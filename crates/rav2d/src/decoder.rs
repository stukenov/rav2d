use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once};

use crate::cpu;
use crate::data::Data;
use crate::headers::{FrameHeader, PixelLayout, SequenceHeader};
use crate::log::{rav2d_log, Logger};
use crate::mem::MemPool;
use crate::picture::{
    DefaultPicAllocator, EventFlags, PicAllocator, Picture, ThreadPicture,
    PICTURE_FLAG_NEW_OP_PARAMS_INFO, PICTURE_FLAG_NEW_SEQUENCE,
};

pub const MAX_THREADS: u32 = 256;
pub const MAX_FRAME_DELAY: u32 = 256;
pub const EOF: i32 = -1;
pub const EAGAIN: i32 = -2;
pub const EINVAL: i32 = -3;
pub const ENOMEM: i32 = -4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InloopFilterType {
    None = 0,
    Deblock = 1,
    Cdef = 2,
    Restoration = 4,
    Wiener = 8,
    Gdf = 16,
    All = 31,
}

impl Default for InloopFilterType {
    fn default() -> Self {
        Self::All
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DecodeFrameType {
    All = 0,
    Reference = 1,
    Intra = 2,
    Key = 3,
}

impl Default for DecodeFrameType {
    fn default() -> Self {
        Self::All
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub n_threads: u32,
    pub max_frame_delay: u32,
    pub apply_grain: bool,
    pub operating_point: u32,
    pub all_layers: bool,
    pub frame_size_limit: u32,
    pub strict_std_compliance: bool,
    pub output_invisible_frames: bool,
    pub inloop_filters: InloopFilterType,
    pub decode_frame_type: DecodeFrameType,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            n_threads: 0,
            max_frame_delay: 0,
            apply_grain: true,
            operating_point: 0,
            all_layers: true,
            frame_size_limit: 0,
            strict_std_compliance: false,
            output_invisible_frames: false,
            inloop_filters: InloopFilterType::All,
            decode_frame_type: DecodeFrameType::All,
        }
    }
}

fn get_num_threads(s: &Settings) -> (u32, u32) {
    #[rustfmt::skip]
    const FC_LUT: [u8; 49] = [
        1,
        2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    ];

    let n_tc = if s.n_threads > 0 {
        s.n_threads.clamp(1, MAX_THREADS)
    } else {
        (cpu::num_logical_processors() as u32).clamp(1, MAX_THREADS)
    };

    let n_fc = if s.max_frame_delay > 0 {
        s.max_frame_delay.min(n_tc)
    } else if n_tc < 50 {
        FC_LUT[(n_tc - 1) as usize] as u32
    } else {
        8
    };

    (n_tc, n_fc)
}

pub fn get_frame_delay(s: &Settings) -> Result<u32, i32> {
    if s.n_threads > MAX_THREADS || s.max_frame_delay > MAX_FRAME_DELAY {
        return Err(EINVAL);
    }
    let (_, n_fc) = get_num_threads(s);
    Ok(n_fc)
}

static INIT_ONCE: Once = Once::new();

fn init_internal() {
    INIT_ONCE.call_once(|| {
        cpu::init_cpu();
    });
}

struct OutputQueue {
    pic: ThreadPicture,
    res: i32,
}

struct RefEntry {
    p: ThreadPicture,
    segmap: Option<Vec<u8>>,
    refmvs: Option<Vec<u8>>,
}

impl RefEntry {
    fn new() -> Self {
        Self {
            p: ThreadPicture::new(),
            segmap: None,
            refmvs: None,
        }
    }
}

pub struct Decoder {
    logger: Logger,
    allocator: Arc<dyn PicAllocator>,
    apply_grain: bool,
    operating_point: u32,
    all_layers: bool,
    frame_size_limit: u32,
    strict_std_compliance: bool,
    output_invisible_frames: bool,
    inloop_filters: InloopFilterType,
    decode_frame_type: DecodeFrameType,

    n_tc: u32,
    n_fc: u32,

    seq_hdr: Option<Arc<SequenceHeader>>,
    frame_hdr: Option<Arc<FrameHeader>>,

    input: Data,
    drain: bool,
    flush: AtomicBool,

    refs: [RefEntry; 8],

    dpb: Vec<OutputQueue>,
    dpb_in: usize,
    dpb_out: usize,
    dpb_sz: usize,

    seq_hdr_pool: Arc<MemPool>,
    frame_hdr_pool: Arc<MemPool>,
    segmap_pool: Arc<MemPool>,
    segmap_uv_pool: Arc<MemPool>,
    refmvs_pool: Arc<MemPool>,
    ccsomap_pool: Arc<MemPool>,
    pic_ctx_pool: Arc<MemPool>,
    cdf_pool: Arc<MemPool>,
    fgm_pool: Arc<MemPool>,
    ci_pool: Arc<MemPool>,
    picture_pool: Arc<MemPool>,

    task_thread: Option<TaskThread>,
}

struct TaskThread {
    lock: Mutex<()>,
    cond: Condvar,
    cur: u32,
    n_passes: u32,
}

impl Decoder {
    pub fn open(s: &Settings) -> Result<Self, i32> {
        init_internal();

        if s.n_threads > MAX_THREADS || s.max_frame_delay > MAX_FRAME_DELAY {
            return Err(EINVAL);
        }
        if s.operating_point > 31 {
            return Err(EINVAL);
        }

        let (n_tc, n_fc) = get_num_threads(s);

        let allocator: Arc<dyn PicAllocator> = Arc::new(DefaultPicAllocator::new());
        let logger = Logger::with_default();

        let dpb_sz = n_fc as usize + 16;
        let mut dpb = Vec::with_capacity(dpb_sz);
        for _ in 0..dpb_sz {
            dpb.push(OutputQueue {
                pic: ThreadPicture::new(),
                res: 0,
            });
        }

        let task_thread = if n_tc > 1 {
            Some(TaskThread {
                lock: Mutex::new(()),
                cond: Condvar::new(),
                cur: n_fc,
                n_passes: 1 + (n_tc > 1) as u32 + (n_fc > 1) as u32,
            })
        } else {
            None
        };

        let refs = std::array::from_fn(|_| RefEntry::new());

        Ok(Self {
            logger,
            allocator,
            apply_grain: s.apply_grain,
            operating_point: s.operating_point,
            all_layers: s.all_layers,
            frame_size_limit: s.frame_size_limit,
            strict_std_compliance: s.strict_std_compliance,
            output_invisible_frames: s.output_invisible_frames,
            inloop_filters: s.inloop_filters,
            decode_frame_type: s.decode_frame_type,
            n_tc,
            n_fc,
            seq_hdr: None,
            frame_hdr: None,
            input: Data::new(),
            drain: false,
            flush: AtomicBool::new(false),
            refs,
            dpb,
            dpb_in: 0,
            dpb_out: 0,
            dpb_sz,
            seq_hdr_pool: Arc::new(MemPool::new()),
            frame_hdr_pool: Arc::new(MemPool::new()),
            segmap_pool: Arc::new(MemPool::new()),
            segmap_uv_pool: Arc::new(MemPool::new()),
            refmvs_pool: Arc::new(MemPool::new()),
            ccsomap_pool: Arc::new(MemPool::new()),
            pic_ctx_pool: Arc::new(MemPool::new()),
            cdf_pool: Arc::new(MemPool::new()),
            fgm_pool: Arc::new(MemPool::new()),
            ci_pool: Arc::new(MemPool::new()),
            picture_pool: Arc::new(MemPool::new()),
            task_thread,
        })
    }

    pub fn send_data(&mut self, data: Option<Data>) -> Result<(), i32> {
        match data {
            None => {
                self.drain = true;
                Ok(())
            }
            Some(d) => {
                if self.drain {
                    return Err(EOF);
                }
                if d.is_empty() || d.len() > usize::MAX / 2 {
                    return Err(EINVAL);
                }
                if self.input.has_data() {
                    return Err(EAGAIN);
                }
                self.input = d;
                Ok(())
            }
        }
    }

    pub fn get_picture(&mut self) -> Result<Picture, i32> {
        self.gen_picture()?;

        if self.drain {
            self.queue_flush();
        }

        self.output_image()
    }

    fn output_picture_ready(&self) -> bool {
        if self.dpb_out == self.dpb_in {
            return false;
        }
        true
    }

    fn gen_picture(&mut self) -> Result<(), i32> {
        if self.output_picture_ready() {
            return Ok(());
        }

        while !self.input.is_empty() {
            // In full implementation, would call obu::parse_obus here
            // For now, consume all input
            let len = self.input.len();
            self.input.consume(len);

            if self.output_picture_ready() {
                break;
            }
        }

        Ok(())
    }

    fn output_image(&mut self) -> Result<Picture, i32> {
        if self.dpb_in == self.dpb_out {
            if !self.drain {
                return Err(EAGAIN);
            }
            self.drain = false;
            return Err(EOF);
        }

        let q = &mut self.dpb[self.dpb_out];
        let mut pic = Picture::new();
        std::mem::swap(&mut pic, &mut q.pic.p);
        q.pic.unref();

        self.dpb_out += 1;
        if self.dpb_out == self.dpb_sz {
            self.dpb_out = 0;
        }

        Ok(pic)
    }

    fn queue_flush(&mut self) {
        // Flush implicit show frames from refs
    }

    pub fn flush(&mut self) {
        self.input.unref();

        for q in &mut self.dpb {
            if q.pic.p.has_data() {
                q.pic.unref();
            }
        }
        self.dpb_in = 0;
        self.dpb_out = 0;
        self.drain = false;

        for r in &mut self.refs {
            if r.p.p.has_data() {
                r.p.unref();
            }
            r.segmap = None;
            r.refmvs = None;
        }

        self.frame_hdr = None;
        self.seq_hdr = None;

        self.flush.store(false, Ordering::Release);
    }

    pub fn n_threads(&self) -> u32 {
        self.n_tc
    }

    pub fn n_frame_contexts(&self) -> u32 {
        self.n_fc
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.flush();

        self.seq_hdr_pool.end();
        self.frame_hdr_pool.end();
        self.segmap_pool.end();
        self.segmap_uv_pool.end();
        self.refmvs_pool.end();
        self.ccsomap_pool.end();
        self.pic_ctx_pool.end();
        self.cdf_pool.end();
        self.fgm_pool.end();
        self.ci_pool.end();
        self.picture_pool.end();
    }
}

pub fn version() -> &'static str {
    "0.1.0"
}

pub fn version_api() -> u32 {
    (0 << 16) | (1 << 8) | 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let s = Settings::default();
        assert_eq!(s.n_threads, 0);
        assert_eq!(s.max_frame_delay, 0);
        assert!(s.apply_grain);
        assert_eq!(s.operating_point, 0);
        assert!(s.all_layers);
    }

    #[test]
    fn test_get_num_threads() {
        let mut s = Settings::default();
        s.n_threads = 4;
        let (n_tc, n_fc) = get_num_threads(&s);
        assert_eq!(n_tc, 4);
        assert_eq!(n_fc, 2);
    }

    #[test]
    fn test_get_num_threads_single() {
        let mut s = Settings::default();
        s.n_threads = 1;
        let (n_tc, n_fc) = get_num_threads(&s);
        assert_eq!(n_tc, 1);
        assert_eq!(n_fc, 1);
    }

    #[test]
    fn test_get_num_threads_many() {
        let mut s = Settings::default();
        s.n_threads = 49;
        let (n_tc, n_fc) = get_num_threads(&s);
        assert_eq!(n_tc, 49);
        assert_eq!(n_fc, 7);
    }

    #[test]
    fn test_get_num_threads_over_50() {
        let mut s = Settings::default();
        s.n_threads = 100;
        let (n_tc, n_fc) = get_num_threads(&s);
        assert_eq!(n_tc, 100);
        assert_eq!(n_fc, 8);
    }

    #[test]
    fn test_get_frame_delay() {
        let mut s = Settings::default();
        s.n_threads = 8;
        assert_eq!(get_frame_delay(&s).unwrap(), 3);
    }

    #[test]
    fn test_get_frame_delay_invalid() {
        let mut s = Settings::default();
        s.n_threads = MAX_THREADS + 1;
        assert_eq!(get_frame_delay(&s), Err(EINVAL));
    }

    #[test]
    fn test_decoder_open() {
        let s = Settings::default();
        let decoder = Decoder::open(&s);
        assert!(decoder.is_ok());
        let d = decoder.unwrap();
        assert!(d.n_threads() >= 1);
    }

    #[test]
    fn test_decoder_open_single_thread() {
        let mut s = Settings::default();
        s.n_threads = 1;
        let d = Decoder::open(&s).unwrap();
        assert_eq!(d.n_threads(), 1);
        assert_eq!(d.n_frame_contexts(), 1);
    }

    #[test]
    fn test_decoder_open_invalid() {
        let mut s = Settings::default();
        s.operating_point = 32;
        assert!(Decoder::open(&s).is_err());
    }

    #[test]
    fn test_send_data_drain() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        assert!(d.send_data(None).is_ok());
        assert!(d.drain);
    }

    #[test]
    fn test_send_data_after_drain() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        d.send_data(None).unwrap();
        let data = Data::wrap(vec![1, 2, 3]);
        assert_eq!(d.send_data(Some(data)), Err(EOF));
    }

    #[test]
    fn test_send_data_empty() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        let data = Data::new();
        assert_eq!(d.send_data(Some(data)), Err(EINVAL));
    }

    #[test]
    fn test_send_data_double() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        d.send_data(Some(Data::wrap(vec![1, 2, 3]))).unwrap();
        assert_eq!(
            d.send_data(Some(Data::wrap(vec![4, 5, 6]))),
            Err(EAGAIN)
        );
    }

    #[test]
    fn test_get_picture_no_data() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        assert_eq!(d.get_picture().err(), Some(EAGAIN));
    }

    #[test]
    fn test_flush() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        d.send_data(Some(Data::wrap(vec![1, 2, 3]))).unwrap();
        d.flush();
        assert!(d.input.is_empty());
        assert!(!d.drain);
    }

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_version_api() {
        assert!(version_api() > 0);
    }

    #[test]
    fn test_decoder_drop() {
        let s = Settings::default();
        let d = Decoder::open(&s).unwrap();
        drop(d);
    }
}
