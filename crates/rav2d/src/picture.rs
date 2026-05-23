use std::ptr::NonNull;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use crate::data::DataProps;
use crate::headers::{FrameHeader, PixelLayout, SequenceHeader};
use crate::mem::MemPool;

pub const PICTURE_ALIGNMENT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PictureParameters {
    pub w: i32,
    pub h: i32,
    pub layout: PixelLayout,
    pub bpc: i32,
}

pub struct PictureData {
    ptr: NonNull<u8>,
    size: usize,
}

unsafe impl Send for PictureData {}
unsafe impl Sync for PictureData {}

impl PictureData {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

pub trait PicAllocator: Send + Sync {
    fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation>;
    fn release_picture(&self, alloc: PictureAllocation);
}

pub struct PictureAllocation {
    pub data: [Option<NonNull<u8>>; 3],
    pub stride: [isize; 2],
    pub allocator_data: Option<NonNull<u8>>,
    pool_size: usize,
}

unsafe impl Send for PictureAllocation {}
unsafe impl Sync for PictureAllocation {}

pub struct DefaultPicAllocator {
    pool: Arc<MemPool>,
}

impl DefaultPicAllocator {
    pub fn new() -> Self {
        Self {
            pool: Arc::new(MemPool::new()),
        }
    }

    pub fn with_pool(pool: Arc<MemPool>) -> Self {
        Self { pool }
    }
}

impl PicAllocator for DefaultPicAllocator {
    fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation> {
        let hbd = p.bpc > 8;
        let aligned_w = (p.w as usize + 127) & !127;
        let aligned_h = (p.h as usize + 127) & !127;
        let has_chroma = p.layout != PixelLayout::I400;
        let ss_ver = p.layout == PixelLayout::I420;
        let ss_hor = p.layout != PixelLayout::I444;

        let mut y_stride = (aligned_w << (hbd as usize)) as isize;
        let mut uv_stride = if has_chroma {
            y_stride >> (ss_hor as usize)
        } else {
            0
        };

        if y_stride & 1023 == 0 {
            y_stride += PICTURE_ALIGNMENT as isize;
        }
        if uv_stride & 1023 == 0 && has_chroma {
            uv_stride += PICTURE_ALIGNMENT as isize;
        }

        let y_sz = y_stride as usize * aligned_h;
        let uv_sz = uv_stride as usize * (aligned_h >> (ss_ver as usize));
        let pic_size = y_sz + 2 * uv_sz;

        let total = pic_size + PICTURE_ALIGNMENT;
        let buf = self.pool.pop(total)?;

        let buf_ptr = buf.as_ptr();
        let data0 = buf;
        let data1 = if has_chroma {
            NonNull::new(unsafe { buf_ptr.add(y_sz) })
        } else {
            None
        };
        let data2 = if has_chroma {
            NonNull::new(unsafe { buf_ptr.add(y_sz + uv_sz) })
        } else {
            None
        };

        Some(PictureAllocation {
            data: [Some(data0), data1, data2],
            stride: [y_stride, uv_stride],
            allocator_data: Some(buf),
            pool_size: total,
        })
    }

    fn release_picture(&self, alloc: PictureAllocation) {
        if let Some(ptr) = alloc.allocator_data {
            self.pool.push(ptr, alloc.pool_size);
        }
    }
}

pub struct Picture {
    pub p: PictureParameters,
    pub data: [Option<NonNull<u8>>; 3],
    pub stride: [isize; 2],
    pub seq_hdr: Option<Arc<SequenceHeader>>,
    pub frame_hdr: Option<Arc<FrameHeader>>,
    pub props: DataProps,
    allocation: Option<PictureAllocation>,
    allocator: Option<Arc<dyn PicAllocator>>,
}

unsafe impl Send for Picture {}
unsafe impl Sync for Picture {}

impl Picture {
    pub fn new() -> Self {
        Self {
            p: PictureParameters {
                w: 0,
                h: 0,
                layout: PixelLayout::I400,
                bpc: 0,
            },
            data: [None, None, None],
            stride: [0, 0],
            seq_hdr: None,
            frame_hdr: None,
            props: DataProps::new(),
            allocation: None,
            allocator: None,
        }
    }

    pub fn alloc(
        w: i32,
        h: i32,
        layout: PixelLayout,
        bpc: i32,
        seq_hdr: Option<Arc<SequenceHeader>>,
        frame_hdr: Option<Arc<FrameHeader>>,
        allocator: Arc<dyn PicAllocator>,
    ) -> Option<Self> {
        let params = PictureParameters { w, h, layout, bpc };
        let alloc = allocator.alloc_picture(&params)?;

        Some(Self {
            p: params,
            data: alloc.data,
            stride: alloc.stride,
            seq_hdr,
            frame_hdr,
            props: DataProps::new(),
            allocation: Some(alloc),
            allocator: Some(allocator),
        })
    }

    pub fn has_data(&self) -> bool {
        self.data[0].is_some()
    }

    pub fn unref(&mut self) {
        if let (Some(alloc), Some(allocator)) = (self.allocation.take(), self.allocator.take()) {
            allocator.release_picture(alloc);
        }
        self.data = [None, None, None];
        self.stride = [0, 0];
        self.seq_hdr = None;
        self.frame_hdr = None;
        self.props = DataProps::new();
        self.p = PictureParameters {
            w: 0,
            h: 0,
            layout: PixelLayout::I400,
            bpc: 0,
        };
    }
}

impl Default for Picture {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        self.unref();
    }
}

impl std::fmt::Debug for Picture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Picture")
            .field("params", &self.p)
            .field("has_data", &self.has_data())
            .finish()
    }
}

pub struct ThreadPicture {
    pub p: Picture,
    pub visible: bool,
    pub showable: bool,
    pub progress: Option<[AtomicI32; 3]>,
    pub flags: u32,
}

impl ThreadPicture {
    pub fn new() -> Self {
        Self {
            p: Picture::new(),
            visible: false,
            showable: false,
            progress: None,
            flags: 0,
        }
    }

    pub fn unref(&mut self) {
        self.p.unref();
        self.progress = None;
    }
}

impl Default for ThreadPicture {
    fn default() -> Self {
        Self::new()
    }
}

pub const PICTURE_FLAG_NEW_SEQUENCE: u32 = 1 << 0;
pub const PICTURE_FLAG_NEW_OP_PARAMS_INFO: u32 = 1 << 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventFlags {
    None,
    NewSequence,
    NewOpParamsInfo,
    Both,
}

impl From<u32> for EventFlags {
    fn from(flags: u32) -> Self {
        match (
            flags & PICTURE_FLAG_NEW_SEQUENCE != 0,
            flags & PICTURE_FLAG_NEW_OP_PARAMS_INFO != 0,
        ) {
            (false, false) => EventFlags::None,
            (true, false) => EventFlags::NewSequence,
            (false, true) => EventFlags::NewOpParamsInfo,
            (true, true) => EventFlags::Both,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pic_alloc_i420() {
        let allocator = DefaultPicAllocator::new();
        let params = PictureParameters {
            w: 1920,
            h: 1080,
            layout: PixelLayout::I420,
            bpc: 8,
        };
        let alloc = allocator.alloc_picture(&params).unwrap();
        assert!(alloc.data[0].is_some());
        assert!(alloc.data[1].is_some());
        assert!(alloc.data[2].is_some());
        assert!(alloc.stride[0] > 0);
        assert!(alloc.stride[1] > 0);
        allocator.release_picture(alloc);
    }

    #[test]
    fn test_default_pic_alloc_i400() {
        let allocator = DefaultPicAllocator::new();
        let params = PictureParameters {
            w: 640,
            h: 480,
            layout: PixelLayout::I400,
            bpc: 8,
        };
        let alloc = allocator.alloc_picture(&params).unwrap();
        assert!(alloc.data[0].is_some());
        assert!(alloc.data[1].is_none());
        assert!(alloc.data[2].is_none());
        allocator.release_picture(alloc);
    }

    #[test]
    fn test_default_pic_alloc_10bpc() {
        let allocator = DefaultPicAllocator::new();
        let params = PictureParameters {
            w: 1920,
            h: 1080,
            layout: PixelLayout::I420,
            bpc: 10,
        };
        let alloc = allocator.alloc_picture(&params).unwrap();
        assert!(alloc.stride[0] > 1920);
        allocator.release_picture(alloc);
    }

    #[test]
    fn test_picture_new_empty() {
        let p = Picture::new();
        assert!(!p.has_data());
    }

    #[test]
    fn test_picture_alloc_and_drop() {
        let allocator = Arc::new(DefaultPicAllocator::new());
        let p = Picture::alloc(
            320,
            240,
            PixelLayout::I420,
            8,
            None,
            None,
            allocator,
        );
        assert!(p.is_some());
        let p = p.unwrap();
        assert!(p.has_data());
    }

    #[test]
    fn test_picture_unref() {
        let allocator = Arc::new(DefaultPicAllocator::new());
        let mut p = Picture::alloc(
            320,
            240,
            PixelLayout::I420,
            8,
            None,
            None,
            allocator,
        )
        .unwrap();
        p.unref();
        assert!(!p.has_data());
    }

    #[test]
    fn test_stride_avoids_power_of_2() {
        let allocator = DefaultPicAllocator::new();
        let params = PictureParameters {
            w: 1024,
            h: 1024,
            layout: PixelLayout::I420,
            bpc: 8,
        };
        let alloc = allocator.alloc_picture(&params).unwrap();
        assert!(alloc.stride[0] & 1023 != 0);
        allocator.release_picture(alloc);
    }

    #[test]
    fn test_event_flags_conversion() {
        assert_eq!(EventFlags::from(0), EventFlags::None);
        assert_eq!(
            EventFlags::from(PICTURE_FLAG_NEW_SEQUENCE),
            EventFlags::NewSequence
        );
        assert_eq!(
            EventFlags::from(PICTURE_FLAG_NEW_OP_PARAMS_INFO),
            EventFlags::NewOpParamsInfo
        );
        assert_eq!(
            EventFlags::from(PICTURE_FLAG_NEW_SEQUENCE | PICTURE_FLAG_NEW_OP_PARAMS_INFO),
            EventFlags::Both
        );
    }

    #[test]
    fn test_thread_picture_new() {
        let tp = ThreadPicture::new();
        assert!(!tp.visible);
        assert!(!tp.showable);
        assert!(tp.progress.is_none());
    }
}
