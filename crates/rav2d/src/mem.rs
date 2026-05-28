use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::Mutex;

const POOL_ALIGNMENT: usize = 64;

struct PoolEntry {
    ptr: NonNull<u8>,
    layout: Layout,
    size: usize,
}

// SAFETY: PoolEntry owns its allocation; access is serialized by the MemPool mutex.
unsafe impl Send for PoolEntry {}

pub struct MemPool {
    inner: Mutex<PoolInner>,
}

struct PoolInner {
    free_list: Vec<PoolEntry>,
    ended: bool,
}

impl Default for MemPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemPool {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PoolInner {
                free_list: Vec::new(),
                ended: false,
            }),
        }
    }

    pub fn pop(&self, size: usize) -> Option<NonNull<u8>> {
        let mut inner = self.inner.lock().unwrap();
        if inner.ended {
            return None;
        }
        if let Some(idx) = inner.free_list.iter().position(|e| e.size == size) {
            let entry = inner.free_list.swap_remove(idx);
            return Some(entry.ptr);
        }
        drop(inner);

        let layout = Layout::from_size_align(size, POOL_ALIGNMENT).ok()?;
        let ptr = unsafe { alloc::alloc(layout) };
        NonNull::new(ptr)
    }

    pub fn push(&self, ptr: NonNull<u8>, size: usize) {
        let mut inner = self.inner.lock().unwrap();
        if inner.ended {
            let layout = Layout::from_size_align(size, POOL_ALIGNMENT).unwrap();
            unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
            return;
        }
        let layout = Layout::from_size_align(size, POOL_ALIGNMENT).unwrap();
        inner.free_list.push(PoolEntry { ptr, layout, size });
    }

    pub fn end(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.ended = true;
        for entry in inner.free_list.drain(..) {
            unsafe { alloc::dealloc(entry.ptr.as_ptr(), entry.layout) };
        }
    }
}

impl Drop for MemPool {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap();
        for entry in inner.free_list.drain(..) {
            unsafe { alloc::dealloc(entry.ptr.as_ptr(), entry.layout) };
        }
    }
}

pub fn alloc_aligned(size: usize, align: usize) -> Option<NonNull<u8>> {
    if size == 0 {
        return None;
    }
    let layout = Layout::from_size_align(size, align).ok()?;
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    NonNull::new(ptr)
}

pub fn free_aligned(ptr: NonNull<u8>, size: usize, align: usize) {
    if let Ok(layout) = Layout::from_size_align(size, align) {
        unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_pop_push() {
        let pool = MemPool::new();
        let ptr = pool.pop(1024).unwrap();
        pool.push(ptr, 1024);
        let ptr2 = pool.pop(1024).unwrap();
        assert_eq!(ptr, ptr2);
        pool.push(ptr2, 1024);
    }

    #[test]
    fn test_pool_different_sizes() {
        let pool = MemPool::new();
        let p1 = pool.pop(128).unwrap();
        let p2 = pool.pop(256).unwrap();
        pool.push(p1, 128);
        pool.push(p2, 256);
        let p3 = pool.pop(256).unwrap();
        assert_eq!(p3, p2);
        let p4 = pool.pop(128).unwrap();
        assert_eq!(p4, p1);
        pool.push(p3, 256);
        pool.push(p4, 128);
    }

    #[test]
    fn test_pool_end() {
        let pool = MemPool::new();
        let ptr = pool.pop(512).unwrap();
        pool.push(ptr, 512);
        pool.end();
        assert!(pool.pop(512).is_none());
    }

    #[test]
    fn test_pool_push_after_end() {
        let pool = MemPool::new();
        let ptr = pool.pop(256).unwrap();
        pool.end();
        pool.push(ptr, 256);
    }

    #[test]
    fn test_alloc_aligned_basic() {
        let ptr = alloc_aligned(1024, 64).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        free_aligned(ptr, 1024, 64);
    }

    #[test]
    fn test_alloc_aligned_zero() {
        assert!(alloc_aligned(0, 64).is_none());
    }
}
