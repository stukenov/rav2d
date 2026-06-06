use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread;

pub const FRAME_ERROR: u32 = u32::MAX - 1;
pub const TILE_ERROR: i32 = i32::MAX - 1;

/// Run a set of independent, disjoint-output jobs, optionally in parallel.
///
/// Each job is a `FnOnce` operating on its own non-overlapping output region
/// (e.g. one colour plane). The jobs are deterministic and do not communicate,
/// so running them on separate threads produces output byte-identical to
/// running them sequentially — only the order of side-effect-free writes to
/// disjoint memory changes.
///
/// `n_threads <= 1` (or a single job) runs everything inline on the caller,
/// keeping the single-thread path literally unchanged (no thread spawn, no
/// scope). With `n_threads > 1` the jobs are dispatched across a scoped thread
/// pool of at most `n_threads` workers; the scope joins all workers before
/// returning, so all writes are visible on return.
pub fn run_disjoint_jobs<'env, F>(n_threads: u32, jobs: Vec<F>)
where
    F: FnOnce() + Send + 'env,
{
    if n_threads <= 1 || jobs.len() <= 1 {
        for job in jobs {
            job();
        }
        return;
    }

    // Cap the worker count at the number of jobs; the caller controls the upper
    // bound via `n_threads`.
    let max_workers = (n_threads as usize).min(jobs.len());
    // Keep one job to run on the calling thread so we use `max_workers` threads
    // total (max_workers-1 spawned + the caller), avoiding an idle main thread.
    let mut jobs = jobs;
    let inline_job = jobs.pop();
    thread::scope(|scope| {
        // Distribute the remaining jobs round-robin onto spawned workers. Each
        // spawned worker owns a disjoint subset of the jobs.
        let n_spawn = (max_workers - 1).min(jobs.len());
        if n_spawn == 0 {
            for job in jobs.drain(..) {
                job();
            }
        } else {
            // Bucket the jobs so each spawned worker gets a contiguous chunk.
            let mut buckets: Vec<Vec<F>> = (0..n_spawn).map(|_| Vec::new()).collect();
            for (i, job) in jobs.drain(..).enumerate() {
                buckets[i % n_spawn].push(job);
            }
            for bucket in buckets {
                scope.spawn(move || {
                    for job in bucket {
                        job();
                    }
                });
            }
        }
        if let Some(job) = inline_job {
            job();
        }
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskType {
    Init = 0,
    InitCdf = 1,
    TileSbrow = 2,
    Deblock = 3,
    Cdef = 4,
    LoopRestoration = 5,
    SuperResolution = 6,
    Reconstruction = 7,
    FilmGrain = 8,
    EntropyCoding = 9,
}

pub struct Task {
    pub task_type: TaskType,
    pub sby: i32,
    pub recon_progress: i32,
    pub deblock_progress: i32,
    next: Option<Box<Task>>,
}

impl Task {
    pub fn new(task_type: TaskType, sby: i32) -> Self {
        Self {
            task_type,
            sby,
            recon_progress: 0,
            deblock_progress: 0,
            next: None,
        }
    }
}

pub struct TaskList {
    head: Option<Box<Task>>,
    tail: *mut Task,
    len: usize,
}

// SAFETY: TaskList owns all nodes via Box; tail pointer only used while head is live.
unsafe impl Send for TaskList {}

impl TaskList {
    pub fn new() -> Self {
        Self {
            head: None,
            tail: std::ptr::null_mut(),
            len: 0,
        }
    }

    pub fn push_back(&mut self, task: Task) {
        let mut boxed = Box::new(task);
        let raw = &mut *boxed as *mut Task;
        if self.tail.is_null() {
            self.head = Some(boxed);
        } else {
            unsafe { (*self.tail).next = Some(boxed) };
        }
        self.tail = raw;
        self.len += 1;
    }

    pub fn pop_front(&mut self) -> Option<Box<Task>> {
        self.head.take().map(|mut node| {
            self.head = node.next.take();
            if self.head.is_none() {
                self.tail = std::ptr::null_mut();
            }
            self.len -= 1;
            node
        })
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Default for TaskList {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PendingTasks {
    list: Mutex<TaskList>,
    merge: AtomicBool,
}

impl Default for PendingTasks {
    fn default() -> Self {
        Self::new()
    }
}

impl PendingTasks {
    pub fn new() -> Self {
        Self {
            list: Mutex::new(TaskList::new()),
            merge: AtomicBool::new(false),
        }
    }

    pub fn add(&self, task: Task) {
        let mut list = self.list.lock().unwrap();
        list.push_back(task);
        self.merge.store(true, Ordering::Release);
    }

    pub fn drain(&self) -> TaskList {
        let mut list = self.list.lock().unwrap();
        self.merge.store(false, Ordering::Release);
        let mut taken = TaskList::new();
        std::mem::swap(&mut *list, &mut taken);
        taken
    }

    pub fn needs_merge(&self) -> bool {
        self.merge.load(Ordering::Acquire)
    }
}

pub struct TaskThreadData {
    pub lock: Mutex<TaskThreadState>,
    pub cond: Condvar,
    pub first: AtomicU32,
    pub cur: AtomicU32,
    pub reset_task_cur: AtomicU32,
    pub cond_signaled: AtomicI32,
    pub n_passes: u32,
}

pub struct TaskThreadState {
    pub die: bool,
}

impl TaskThreadData {
    pub fn new(n_passes: u32) -> Self {
        Self {
            lock: Mutex::new(TaskThreadState { die: false }),
            cond: Condvar::new(),
            first: AtomicU32::new(0),
            cur: AtomicU32::new(0),
            reset_task_cur: AtomicU32::new(u32::MAX),
            cond_signaled: AtomicI32::new(0),
            n_passes,
        }
    }

    pub fn signal(&self) {
        self.cond_signaled.fetch_add(1, Ordering::Release);
        self.cond.notify_one();
    }

    pub fn broadcast(&self) {
        self.cond_signaled.fetch_add(1, Ordering::Release);
        self.cond.notify_all();
    }
}

pub struct FrameThreadData {
    pub tasks: TaskList,
    pub pending: PendingTasks,
    pub lock: Mutex<FrameThreadState>,
    pub cond: Condvar,
    pub retval: AtomicI32,
    pub error: AtomicBool,
}

pub struct FrameThreadState {
    pub n_tile_data: u32,
    pub scheduled: bool,
}

impl FrameThreadData {
    pub fn new() -> Self {
        Self {
            tasks: TaskList::new(),
            pending: PendingTasks::new(),
            lock: Mutex::new(FrameThreadState {
                n_tile_data: 0,
                scheduled: false,
            }),
            cond: Condvar::new(),
            retval: AtomicI32::new(0),
            error: AtomicBool::new(false),
        }
    }
}

impl Default for FrameThreadData {
    fn default() -> Self {
        Self::new()
    }
}

pub struct WorkerHandle {
    handle: Option<thread::JoinHandle<()>>,
    pub flushed: AtomicBool,
    pub die: AtomicBool,
}

impl WorkerHandle {
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            handle: Some(thread::spawn(f)),
            flushed: AtomicBool::new(false),
            die: AtomicBool::new(false),
        }
    }

    pub fn join(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.join();
    }
}

pub fn create_tile_sbrow_tasks(n_tiles: usize, sb_rows: usize, _pass: i32) -> Vec<Task> {
    let mut tasks = Vec::with_capacity(n_tiles * sb_rows);
    for _tile in 0..n_tiles {
        for sby in 0..sb_rows as i32 {
            tasks.push(Task::new(TaskType::TileSbrow, sby));
        }
    }
    tasks
}

pub fn create_filter_sbrow_tasks(sb_rows: usize) -> Vec<Task> {
    let mut tasks = Vec::new();
    for sby in 0..sb_rows as i32 {
        tasks.push(Task::new(TaskType::Deblock, sby));
        tasks.push(Task::new(TaskType::Cdef, sby));
        tasks.push(Task::new(TaskType::LoopRestoration, sby));
    }
    tasks
}

pub fn frame_init_task() -> Task {
    Task::new(TaskType::Init, 0)
}

pub fn frame_init_cdf_task() -> Task {
    Task::new(TaskType::InitCdf, 0)
}

pub fn abort_frame(ftd: &FrameThreadData) {
    ftd.error.store(true, Ordering::Release);
    ftd.retval.store(-1, Ordering::Release);
}

pub fn get_frame_progress(progress: &[AtomicI32], idx: usize) -> i32 {
    progress[idx].load(Ordering::Acquire)
}

pub fn set_frame_progress(progress: &[AtomicI32], idx: usize, val: i32) {
    progress[idx].store(val, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_list_push_pop() {
        let mut list = TaskList::new();
        assert!(list.is_empty());
        list.push_back(Task::new(TaskType::Init, 0));
        list.push_back(Task::new(TaskType::Deblock, 1));
        assert_eq!(list.len(), 2);
        let t1 = list.pop_front().unwrap();
        assert_eq!(t1.task_type, TaskType::Init);
        assert_eq!(t1.sby, 0);
        let t2 = list.pop_front().unwrap();
        assert_eq!(t2.task_type, TaskType::Deblock);
        assert_eq!(t2.sby, 1);
        assert!(list.is_empty());
        assert!(list.pop_front().is_none());
    }

    #[test]
    fn test_pending_tasks() {
        let pending = PendingTasks::new();
        assert!(!pending.needs_merge());
        pending.add(Task::new(TaskType::Cdef, 0));
        assert!(pending.needs_merge());
        let drained = pending.drain();
        assert_eq!(drained.len(), 1);
        assert!(!pending.needs_merge());
    }

    #[test]
    fn test_task_thread_data() {
        let ttd = TaskThreadData::new(2);
        assert_eq!(ttd.n_passes, 2);
        assert_eq!(ttd.first.load(Ordering::Relaxed), 0);
        assert_eq!(ttd.reset_task_cur.load(Ordering::Relaxed), u32::MAX);
    }

    #[test]
    fn test_create_tile_sbrow_tasks() {
        let tasks = create_tile_sbrow_tasks(2, 4, 0);
        assert_eq!(tasks.len(), 8);
        for t in &tasks {
            assert_eq!(t.task_type, TaskType::TileSbrow);
        }
    }

    #[test]
    fn test_create_filter_sbrow_tasks() {
        let tasks = create_filter_sbrow_tasks(3);
        assert_eq!(tasks.len(), 9);
        assert_eq!(tasks[0].task_type, TaskType::Deblock);
        assert_eq!(tasks[1].task_type, TaskType::Cdef);
        assert_eq!(tasks[2].task_type, TaskType::LoopRestoration);
    }

    #[test]
    fn test_frame_init_task() {
        let t = frame_init_task();
        assert_eq!(t.task_type, TaskType::Init);
        assert_eq!(t.sby, 0);
    }

    #[test]
    fn test_frame_thread_data() {
        let ftd = FrameThreadData::new();
        assert!(!ftd.error.load(Ordering::Relaxed));
        assert_eq!(ftd.retval.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_abort_frame() {
        let ftd = FrameThreadData::new();
        abort_frame(&ftd);
        assert!(ftd.error.load(Ordering::Relaxed));
        assert_eq!(ftd.retval.load(Ordering::Relaxed), -1);
    }

    #[test]
    fn test_frame_progress() {
        let progress = [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)];
        set_frame_progress(&progress, 1, 42);
        assert_eq!(get_frame_progress(&progress, 1), 42);
    }

    #[test]
    fn test_run_disjoint_jobs_inline() {
        use std::sync::{Arc, Mutex};
        // n_threads == 1 runs inline, in order.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
        for i in 0..3 {
            let log = log.clone();
            jobs.push(Box::new(move || log.lock().unwrap().push(i)));
        }
        run_disjoint_jobs(1, jobs);
        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_run_disjoint_jobs_parallel() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        // Each job writes to its own disjoint slot; result is deterministic
        // regardless of execution order or thread count.
        let counter = Arc::new(AtomicUsize::new(0));
        let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
        for _ in 0..8 {
            let counter = counter.clone();
            jobs.push(Box::new(move || {
                counter.fetch_add(1, Ordering::Relaxed);
            }));
        }
        run_disjoint_jobs(4, jobs);
        assert_eq!(counter.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn test_run_disjoint_jobs_disjoint_writes() {
        // Three disjoint output buffers written by three jobs; parallel result
        // must equal the sequential result byte-for-byte.
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        let mut c = [0u8; 4];
        {
            let (ra, rb, rc) = (&mut a, &mut b, &mut c);
            let jobs: Vec<Box<dyn FnOnce() + Send>> = vec![
                Box::new(move || ra.iter_mut().for_each(|x| *x = 1)),
                Box::new(move || rb.iter_mut().for_each(|x| *x = 2)),
                Box::new(move || rc.iter_mut().for_each(|x| *x = 3)),
            ];
            run_disjoint_jobs(4, jobs);
        }
        assert_eq!(a, [1; 4]);
        assert_eq!(b, [2; 4]);
        assert_eq!(c, [3; 4]);
    }

    #[test]
    fn test_worker_handle_spawn_join() {
        use std::sync::{Arc, atomic::AtomicBool};
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let mut handle = WorkerHandle::spawn(move || {
            done_clone.store(true, Ordering::Relaxed);
        });
        handle.join();
        assert!(done.load(Ordering::Relaxed));
    }
}
