use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::mem::MemPool;

type FreeCallback = Box<dyn Fn() + Send + Sync>;

pub struct Ref {
    data: Option<NonNull<u8>>,
    ref_cnt: AtomicUsize,
    free_callback: FreeCallback,
}

// SAFETY: Ref uses atomic reference counting; data pointer is owned and not aliased.
unsafe impl Send for Ref {}
unsafe impl Sync for Ref {}

impl Ref {
    pub fn create(size: usize) -> Option<Box<Self>> {
        let ptr = crate::mem::alloc_aligned(size, 64)?;
        let addr = ptr.as_ptr() as usize;
        Some(Box::new(Self {
            data: Some(ptr),
            ref_cnt: AtomicUsize::new(1),
            free_callback: Box::new(move || {
                unsafe {
                    crate::mem::free_aligned(NonNull::new_unchecked(addr as *mut u8), size, 64)
                };
            }),
        }))
    }

    pub fn create_from_pool(pool: &Arc<MemPool>, size: usize) -> Option<Box<Self>> {
        let ptr = pool.pop(size)?;
        let pool = Arc::clone(pool);
        let addr = ptr.as_ptr() as usize;
        Some(Box::new(Self {
            data: Some(ptr),
            ref_cnt: AtomicUsize::new(1),
            free_callback: Box::new(move || {
                // SAFETY: addr was obtained from pool.pop and is still valid.
                pool.push(unsafe { NonNull::new_unchecked(addr as *mut u8) }, size);
            }),
        }))
    }

    pub fn wrap(data: NonNull<u8>, free_callback: FreeCallback) -> Box<Self> {
        Box::new(Self {
            data: Some(data),
            ref_cnt: AtomicUsize::new(1),
            free_callback,
        })
    }

    pub fn data(&self) -> Option<NonNull<u8>> {
        self.data
    }

    pub fn inc(&self) {
        self.ref_cnt.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec(&self) -> bool {
        self.ref_cnt.fetch_sub(1, Ordering::Release) == 1
    }

    pub fn is_writable(&self) -> bool {
        self.ref_cnt.load(Ordering::Acquire) == 1 && self.data.is_some()
    }

    pub fn ref_count(&self) -> usize {
        self.ref_cnt.load(Ordering::Relaxed)
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        (self.free_callback)();
    }
}

pub struct SharedRef {
    inner: Option<Box<Ref>>,
}

impl SharedRef {
    pub fn new(r: Box<Ref>) -> Self {
        Self { inner: Some(r) }
    }

    pub fn empty() -> Self {
        Self { inner: None }
    }

    pub fn is_some(&self) -> bool {
        self.inner.is_some()
    }

    pub fn data(&self) -> Option<NonNull<u8>> {
        self.inner.as_ref().and_then(|r| r.data())
    }

    pub fn inc(&self) {
        if let Some(ref r) = self.inner {
            r.inc();
        }
    }

    pub fn is_writable(&self) -> bool {
        self.inner.as_ref().is_some_and(|r| r.is_writable())
    }

    pub fn take(&mut self) -> Option<Box<Ref>> {
        self.inner.take()
    }
}

impl Default for SharedRef {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_create() {
        let r = Ref::create(1024).unwrap();
        assert!(r.data().is_some());
        assert_eq!(r.ref_count(), 1);
        assert!(r.is_writable());
    }

    #[test]
    fn test_ref_inc_dec() {
        let r = Ref::create(256).unwrap();
        r.inc();
        assert_eq!(r.ref_count(), 2);
        assert!(!r.is_writable());
        assert!(!r.dec());
        assert_eq!(r.ref_count(), 1);
        assert!(r.is_writable());
    }

    #[test]
    fn test_ref_dec_to_zero() {
        let r = Ref::create(128).unwrap();
        assert!(r.dec());
    }

    #[test]
    fn test_shared_ref_empty() {
        let sr = SharedRef::empty();
        assert!(!sr.is_some());
        assert!(sr.data().is_none());
        assert!(!sr.is_writable());
    }

    #[test]
    fn test_shared_ref_with_data() {
        let r = Ref::create(512).unwrap();
        let sr = SharedRef::new(r);
        assert!(sr.is_some());
        assert!(sr.data().is_some());
        assert!(sr.is_writable());
    }

    #[test]
    fn test_ref_wrap() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let freed = Arc::new(AtomicBool::new(false));
        let freed_clone = freed.clone();
        let ptr = crate::mem::alloc_aligned(64, 64).unwrap();
        let addr = ptr.as_ptr() as usize;
        let r = Ref::wrap(
            ptr,
            Box::new(move || {
                freed_clone.store(true, Ordering::Relaxed);
                unsafe {
                    crate::mem::free_aligned(NonNull::new_unchecked(addr as *mut u8), 64, 64)
                };
            }),
        );
        assert!(!freed.load(Ordering::Relaxed));
        drop(r);
        assert!(freed.load(Ordering::Relaxed));
    }
}
