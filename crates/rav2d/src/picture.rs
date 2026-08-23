use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;

use crate::data::DataProps;
use crate::headers::{FilmGrainData, FrameHeader, PixelLayout, SequenceHeader};
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

// SAFETY: PictureData owns its allocation exclusively; the NonNull pointer is not aliased.
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

/// Supplies the pixel buffers the decoder writes reconstructed frames into
/// (dav2d's `Dav2dPicAllocator`).
///
/// # Safety
///
/// [`Picture`]'s safe plane accessors ([`Picture::plane_row_bytes`] and
/// friends) hand out `&[u8]`/`&[u16]` views computed directly from the pointers
/// and strides returned here, so an implementation that lies about its
/// allocation makes those safe methods unsound. An implementor must guarantee,
/// for every [`PictureAllocation`] returned by `alloc_picture`:
///
/// * For each plane `pl` whose `data[pl]` is `Some(ptr)`, the range of bytes
///   `ptr.offset(y * stride) .. ptr.offset(y * stride) + row_bytes` is part of
///   a single live allocation for every row `y` in `0..plane_height(pl)`, where
///   `row_bytes = plane_width(pl) * bytes_per_sample`. This must hold for a
///   negative `stride` too (the pointer then addresses the last row).
///   `plane_width`/`plane_height` follow the chroma subsampling of `p.layout`:
///   `(w + ss_hor) >> ss_hor` by `(h + ss_ver) >> ss_ver`.
/// * `data[0]` is `Some` whenever the allocation succeeds; `data[1]`/`data[2]`
///   are `Some` for every layout except `I400`.
/// * The allocation stays live and is not aliased by anything else until the
///   matching `release_picture` call.
/// * For `p.bpc > 8` the plane pointers and byte strides are 2-byte aligned, so
///   rows can be viewed as `u16` samples.
///
/// The decoder additionally requires the extra row/column padding documented on
/// [`DefaultPicAllocator::alloc_picture`]; a custom allocator that under-pads
/// will be caught by the decoder's own bounds checks rather than by these
/// accessors.
pub unsafe trait PicAllocator: Send + Sync {
    fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation>;
    fn release_picture(&self, alloc: PictureAllocation);
}

pub struct PictureAllocation {
    pub data: [Option<NonNull<u8>>; 3],
    pub stride: [isize; 2],
    pub allocator_data: Option<NonNull<u8>>,
    /// Block size handed back to the internal pool by [`DefaultPicAllocator`].
    /// Zero for allocations built through [`PictureAllocation::new`], which the
    /// implementing allocator frees itself.
    pool_size: usize,
}

impl PictureAllocation {
    /// Describe a picture buffer supplied by a custom [`PicAllocator`].
    ///
    /// `allocator_data` is carried through untouched and handed back to
    /// `release_picture`, so an allocator can stash whatever handle it needs to
    /// free the buffer.
    pub fn new(
        data: [Option<NonNull<u8>>; 3],
        stride: [isize; 2],
        allocator_data: Option<NonNull<u8>>,
    ) -> Self {
        Self {
            data,
            stride,
            allocator_data,
            pool_size: 0,
        }
    }
}

// SAFETY: PictureAllocation owns its plane pointers; they are not aliased across threads.
unsafe impl Send for PictureAllocation {}
unsafe impl Sync for PictureAllocation {}

pub struct DefaultPicAllocator {
    pool: Arc<MemPool>,
}

impl Default for DefaultPicAllocator {
    fn default() -> Self {
        Self::new()
    }
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

// SAFETY: every plane is carved out of one `pic_size + PICTURE_ALIGNMENT` byte
// pool block. Strides are positive and each plane reserves `stride * aligned_h`
// bytes with `aligned_w >= w` and `aligned_h >= h` rounded up to 128, so every
// row `y < plane_height` is fully inside the block. Chroma pointers are `Some`
// for every layout but I400. For HBD the strides are `aligned_w << 1` (or that
// plus the 64-byte alignment bump), hence even, and the pool block is 64-byte
// aligned, so `u16` views are aligned.
unsafe impl PicAllocator for DefaultPicAllocator {
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

/// A decoded video frame with pixel data and associated metadata.
///
/// Pixels are read through the safe plane accessors — [`Picture::plane_row_u8`],
/// [`Picture::plane_row_u16`], [`Picture::plane_row_bytes`] and their `rows_*`
/// iterators — which bounds-check the plane index and row number and hand back
/// ordinary slices. The raw pointer and stride are still reachable through
/// [`Picture::plane_ptr`] / [`Picture::plane_stride_bytes`] for FFI callers that
/// need them.
pub struct Picture {
    pub p: PictureParameters,
    pub(crate) data: [Option<NonNull<u8>>; 3],
    pub(crate) stride: [isize; 2],
    pub seq_hdr: Option<Arc<SequenceHeader>>,
    pub frame_hdr: Option<Arc<FrameHeader>>,
    /// Film-grain synthesis parameters for this frame (the selected `c.fgm[id]`
    /// entry), or `None` when the frame carries no grain. Mirrors dav2d's
    /// `Dav2dPicture.fgm`; used at output time to apply grain to a display copy.
    pub fgm: Option<FilmGrainData>,
    pub props: DataProps,
    allocation: Option<PictureAllocation>,
    allocator: Option<Arc<dyn PicAllocator>>,
}

// SAFETY: Picture owns its pixel data via PicAllocator; no mutable aliasing across threads.
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
            fgm: None,
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
            fgm: None,
            props: DataProps::new(),
            allocation: Some(alloc),
            allocator: Some(allocator),
        })
    }

    pub fn has_data(&self) -> bool {
        self.data[0].is_some()
    }

    /// True when this picture stores samples as 16-bit (`bpc > 8`). The plane
    /// allocation already reserves `width << hbd` bytes per row (see
    /// `DefaultPicAllocator::alloc_picture`), so high-bit-depth planes hold two
    /// bytes per sample.
    #[inline]
    pub fn is_hbd(&self) -> bool {
        self.p.bpc > 8
    }

    /// Number of storage bytes per sample (1 for 8bpc, 2 for HBD).
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        if self.is_hbd() { 2 } else { 1 }
    }

    /// Row stride for plane `pl` expressed in **samples** (not bytes). Plane 0
    /// is luma; planes 1/2 are chroma. For the byte stride use
    /// [`Picture::plane_stride_bytes`].
    #[inline]
    pub fn stride_px(&self, pl: usize) -> usize {
        let s = self.stride[if pl == 0 { 0 } else { 1 }].unsigned_abs();
        s / self.bytes_per_sample()
    }

    /// Signed row stride of plane `pl` in bytes. Negative for a bottom-up
    /// allocation, in which case [`Picture::plane_ptr`] addresses the last row.
    /// Planes 1 and 2 share one stride.
    #[inline]
    pub fn plane_stride_bytes(&self, pl: usize) -> isize {
        self.stride[if pl == 0 { 0 } else { 1 }]
    }

    /// Base pointer of plane `pl`, or `None` when the plane is absent (any
    /// chroma plane of an `I400` frame, or an unallocated picture).
    ///
    /// Prefer the row accessors; this exists for FFI callers that must pass the
    /// plane on to C. Reading through the pointer is `unsafe` and bounded by
    /// [`Picture::plane_dimensions`] and [`Picture::plane_stride_bytes`].
    #[inline]
    pub fn plane_ptr(&self, pl: usize) -> Option<NonNull<u8>> {
        *self.data.get(pl)?
    }

    /// Chroma subsampling `(ss_hor, ss_ver)` of this frame's layout.
    #[inline]
    fn subsampling(&self) -> (i32, i32) {
        match self.p.layout {
            PixelLayout::I420 => (1, 1),
            PixelLayout::I422 => (1, 0),
            PixelLayout::I444 | PixelLayout::I400 => (0, 0),
        }
    }

    /// Size of plane `pl` in **samples**, as `(width, height)`.
    ///
    /// Luma is `(w, h)`; chroma is subsampled per the frame's layout. Returns
    /// `None` for a plane this picture does not have — chroma of an `I400`
    /// frame, a `pl` above 2, or a picture with no pixel data.
    pub fn plane_dimensions(&self, pl: usize) -> Option<(usize, usize)> {
        if pl > 2 || self.data.get(pl).copied().flatten().is_none() {
            return None;
        }
        if pl == 0 {
            return Some((self.p.w as usize, self.p.h as usize));
        }
        let (ss_hor, ss_ver) = self.subsampling();
        Some((
            ((self.p.w + ss_hor) >> ss_hor) as usize,
            ((self.p.h + ss_ver) >> ss_ver) as usize,
        ))
    }

    /// Row `y` of plane `pl` as raw sample bytes, without the row padding.
    ///
    /// The slice is `plane_width * bytes_per_sample` long, so for a high-bit-depth
    /// frame it holds native-endian `u16` samples — use [`Picture::plane_row_u16`]
    /// to read those as numbers. Returns `None` if the plane is absent or `y` is
    /// past the end of it.
    pub fn plane_row_bytes(&self, pl: usize, y: usize) -> Option<&[u8]> {
        let (pw, ph) = self.plane_dimensions(pl)?;
        if y >= ph {
            return None;
        }
        let base = self.data[pl]?.as_ptr();
        let row_bytes = pw.checked_mul(self.bytes_per_sample())?;
        let off = (y as isize).checked_mul(self.plane_stride_bytes(pl))?;
        // SAFETY: `PicAllocator` (an unsafe trait) guarantees that every row
        // `y < ph` of an allocated plane covers `row_bytes` valid bytes at
        // `base + y * stride`, for either sign of the stride. `&self` keeps the
        // allocation alive and rules out concurrent mutation.
        Some(unsafe { std::slice::from_raw_parts(base.offset(off), row_bytes) })
    }

    /// Row `y` of plane `pl` as 8-bit samples.
    ///
    /// Returns `None` for a high-bit-depth picture (use [`Picture::plane_row_u16`]),
    /// an absent plane, or a `y` past the end of the plane.
    pub fn plane_row_u8(&self, pl: usize, y: usize) -> Option<&[u8]> {
        if self.is_hbd() {
            return None;
        }
        self.plane_row_bytes(pl, y)
    }

    /// Row `y` of plane `pl` as 16-bit samples, for a high-bit-depth picture.
    ///
    /// Samples occupy the low `bpc` bits of each `u16`. Returns `None` for an
    /// 8-bit picture (use [`Picture::plane_row_u8`]), an absent plane, or a `y`
    /// past the end of the plane.
    pub fn plane_row_u16(&self, pl: usize, y: usize) -> Option<&[u16]> {
        if !self.is_hbd() {
            return None;
        }
        let bytes = self.plane_row_bytes(pl, y)?;
        // The allocator contract promises 2-byte-aligned HBD planes and strides;
        // check anyway so a buggy custom allocator yields `None` and not UB.
        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()) {
            return None;
        }
        // SAFETY: `bytes` is `pw * 2` bytes of a live plane, just checked to be
        // `u16`-aligned; `u16` has no invalid bit patterns and the borrow is
        // tied to the same `&self`.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2) })
    }

    /// Every row of plane `pl` top to bottom, as raw sample bytes. Empty when
    /// the plane is absent.
    pub fn plane_rows_bytes(&self, pl: usize) -> impl Iterator<Item = &[u8]> {
        let ph = self.plane_dimensions(pl).map_or(0, |(_, h)| h);
        (0..ph).filter_map(move |y| self.plane_row_bytes(pl, y))
    }

    /// Every row of plane `pl` top to bottom, as 8-bit samples. Empty when the
    /// plane is absent or the picture is high-bit-depth.
    pub fn plane_rows_u8(&self, pl: usize) -> impl Iterator<Item = &[u8]> {
        let ph = if self.is_hbd() {
            0
        } else {
            self.plane_dimensions(pl).map_or(0, |(_, h)| h)
        };
        (0..ph).filter_map(move |y| self.plane_row_u8(pl, y))
    }

    /// Every row of plane `pl` top to bottom, as 16-bit samples. Empty when the
    /// plane is absent or the picture is 8-bit.
    pub fn plane_rows_u16(&self, pl: usize) -> impl Iterator<Item = &[u16]> {
        let ph = if self.is_hbd() {
            self.plane_dimensions(pl).map_or(0, |(_, h)| h)
        } else {
            0
        };
        (0..ph).filter_map(move |y| self.plane_row_u16(pl, y))
    }

    pub fn unref(&mut self) {
        if let (Some(alloc), Some(allocator)) = (self.allocation.take(), self.allocator.take()) {
            allocator.release_picture(alloc);
        }
        self.data = [None, None, None];
        self.stride = [0, 0];
        self.seq_hdr = None;
        self.frame_hdr = None;
        self.fgm = None;
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
        let p = Picture::alloc(320, 240, PixelLayout::I420, 8, None, None, allocator);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!(p.has_data());
    }

    #[test]
    fn test_picture_unref() {
        let allocator = Arc::new(DefaultPicAllocator::new());
        let mut p = Picture::alloc(320, 240, PixelLayout::I420, 8, None, None, allocator).unwrap();
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

    // -----------------------------------------------------------------------
    // Safe plane accessors
    // -----------------------------------------------------------------------

    fn alloc_pic(w: i32, h: i32, layout: PixelLayout, bpc: i32) -> Picture {
        let allocator = Arc::new(DefaultPicAllocator::new());
        Picture::alloc(w, h, layout, bpc, None, None, allocator).unwrap()
    }

    /// Fill every plane's full stride (padding included) with `fill`, then write
    /// `value(pl, y, x)` over the `pw` visible samples of each row.
    fn paint(pic: &mut Picture, fill: u8, value: impl Fn(usize, usize, usize) -> u8) {
        for pl in 0..3 {
            let Some((pw, ph)) = pic.plane_dimensions(pl) else {
                continue;
            };
            let stride = pic.plane_stride_bytes(pl).unsigned_abs();
            let base = pic.plane_ptr(pl).unwrap().as_ptr();
            let bps = pic.bytes_per_sample();
            for y in 0..ph {
                // SAFETY: the default allocator reserves `stride` bytes for each
                // of the `ph` rows of this plane.
                let row = unsafe { std::slice::from_raw_parts_mut(base.add(y * stride), stride) };
                row.fill(fill);
                for x in 0..pw * bps {
                    row[x] = value(pl, y, x);
                }
            }
        }
    }

    #[test]
    fn test_plane_dimensions_per_layout() {
        // Odd sizes so the chroma rounding is exercised.
        let p420 = alloc_pic(65, 33, PixelLayout::I420, 8);
        assert_eq!(p420.plane_dimensions(0), Some((65, 33)));
        assert_eq!(p420.plane_dimensions(1), Some((33, 17)));
        assert_eq!(p420.plane_dimensions(2), Some((33, 17)));

        let p422 = alloc_pic(65, 33, PixelLayout::I422, 8);
        assert_eq!(p422.plane_dimensions(1), Some((33, 33)));

        let p444 = alloc_pic(65, 33, PixelLayout::I444, 8);
        assert_eq!(p444.plane_dimensions(1), Some((65, 33)));

        let p400 = alloc_pic(65, 33, PixelLayout::I400, 8);
        assert_eq!(p400.plane_dimensions(0), Some((65, 33)));
        assert_eq!(p400.plane_dimensions(1), None, "I400 has no chroma");
        assert_eq!(p400.plane_dimensions(2), None);
    }

    #[test]
    fn test_plane_dimensions_out_of_range() {
        let pic = alloc_pic(32, 32, PixelLayout::I420, 8);
        assert_eq!(pic.plane_dimensions(3), None);
        assert_eq!(pic.plane_dimensions(usize::MAX), None);
        assert_eq!(Picture::new().plane_dimensions(0), None, "no pixel data");
    }

    #[test]
    fn test_plane_row_u8_reads_back_without_padding() {
        let mut pic = alloc_pic(37, 9, PixelLayout::I420, 8);
        // 0xEE marks the row padding: it must never appear in a row slice.
        paint(&mut pic, 0xEE, |pl, y, x| (pl * 50 + y * 5 + x) as u8);

        for pl in 0..3 {
            let (pw, ph) = pic.plane_dimensions(pl).unwrap();
            for y in 0..ph {
                let row = pic.plane_row_u8(pl, y).unwrap();
                assert_eq!(row.len(), pw, "plane {pl} row {y} length");
                for (x, &got) in row.iter().enumerate() {
                    assert_eq!(got, (pl * 50 + y * 5 + x) as u8, "plane {pl} at ({x},{y})");
                }
            }
            assert_eq!(pic.plane_row_u8(pl, ph), None, "one row past the end");
            assert_eq!(pic.plane_row_u8(pl, usize::MAX), None);
        }
        assert_eq!(pic.plane_row_u8(3, 0), None, "no plane 3");
    }

    #[test]
    fn test_plane_rows_iterators_cover_the_plane() {
        let mut pic = alloc_pic(16, 12, PixelLayout::I420, 8);
        paint(&mut pic, 0xEE, |_, y, _| y as u8);

        for pl in 0..3 {
            let (pw, ph) = pic.plane_dimensions(pl).unwrap();
            let rows: Vec<_> = pic.plane_rows_u8(pl).collect();
            assert_eq!(rows.len(), ph);
            for (y, row) in rows.iter().enumerate() {
                assert_eq!(row.len(), pw);
                assert!(row.iter().all(|&v| v == y as u8));
            }
            assert_eq!(pic.plane_rows_bytes(pl).count(), ph);
            assert_eq!(pic.plane_rows_u16(pl).count(), 0, "8bpc has no u16 view");
        }
        assert_eq!(pic.plane_rows_u8(3).count(), 0);
    }

    #[test]
    fn test_plane_row_u16_hbd() {
        let mut pic = alloc_pic(20, 6, PixelLayout::I420, 10);
        assert!(pic.is_hbd());
        assert_eq!(pic.bytes_per_sample(), 2);
        // Little-endian 10-bit samples: 0x0300 | x, i.e. bytes [x, 3].
        paint(
            &mut pic,
            0xEE,
            |_, _, x| if x % 2 == 0 { x as u8 } else { 3 },
        );

        for pl in 0..3 {
            let (pw, ph) = pic.plane_dimensions(pl).unwrap();
            for y in 0..ph {
                let row = pic.plane_row_u16(pl, y).unwrap();
                assert_eq!(row.len(), pw, "plane {pl} row {y} sample count");
                for (i, &s) in row.iter().enumerate() {
                    assert_eq!(s, u16::from_le_bytes([(i * 2) as u8, 3]));
                }
                assert_eq!(
                    pic.plane_row_bytes(pl, y).unwrap().len(),
                    pw * 2,
                    "byte view is twice the sample count"
                );
            }
            assert_eq!(pic.plane_row_u16(pl, ph), None);
            assert_eq!(pic.plane_row_u8(pl, 0), None, "u8 view is 8bpc-only");
            assert_eq!(pic.plane_rows_u8(pl).count(), 0);
            assert_eq!(pic.plane_rows_u16(pl).count(), ph);
        }
    }

    #[test]
    fn test_plane_accessors_on_empty_picture() {
        let pic = Picture::new();
        assert_eq!(pic.plane_row_bytes(0, 0), None);
        assert_eq!(pic.plane_row_u8(0, 0), None);
        assert_eq!(pic.plane_row_u16(0, 0), None);
        assert_eq!(pic.plane_ptr(0), None);
        assert_eq!(pic.plane_rows_bytes(0).count(), 0);
    }

    /// A bottom-up allocator: `data` addresses the last row and the stride is
    /// negative, the layout a capture card or a flipped GL readback would hand
    /// over. The row accessors must still walk top to bottom.
    #[test]
    fn test_plane_rows_with_negative_stride() {
        struct FlippedAllocator;

        // SAFETY: each plane gets its own `stride * ph` byte `Vec`; the pointer
        // handed out addresses the last row and the stride is negative, so
        // `ptr + y * stride` stays inside that buffer for every `y < ph`.
        unsafe impl PicAllocator for FlippedAllocator {
            fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation> {
                let mut data = [None, None, None];
                let mut boxed: Vec<Box<[u8]>> = Vec::new();
                let stride = p.w as usize;
                for (pl, slot) in data.iter_mut().enumerate() {
                    let (pw, ph) = if pl == 0 {
                        (p.w as usize, p.h as usize)
                    } else {
                        ((p.w as usize).div_ceil(2), (p.h as usize).div_ceil(2))
                    };
                    let mut buf = vec![0u8; stride * ph].into_boxed_slice();
                    // Row y holds the byte `y`, written bottom-up.
                    for y in 0..ph {
                        buf[(ph - 1 - y) * stride..(ph - 1 - y) * stride + pw].fill(y as u8);
                    }
                    let last_row = buf[(ph - 1) * stride..].as_mut_ptr();
                    *slot = NonNull::new(last_row);
                    boxed.push(buf);
                }
                // Keep the buffers alive; released in `release_picture`.
                let keep = Box::into_raw(Box::new(boxed)) as *mut u8;
                Some(PictureAllocation::new(
                    data,
                    [-(stride as isize), -(stride as isize)],
                    NonNull::new(keep),
                ))
            }

            fn release_picture(&self, alloc: PictureAllocation) {
                if let Some(ptr) = alloc.allocator_data {
                    // SAFETY: the pointer came from `Box::into_raw` above and is
                    // released exactly once.
                    drop(unsafe { Box::from_raw(ptr.as_ptr() as *mut Vec<Box<[u8]>>) });
                }
            }
        }

        let pic = Picture::alloc(
            8,
            4,
            PixelLayout::I420,
            8,
            None,
            None,
            Arc::new(FlippedAllocator),
        )
        .unwrap();

        assert!(pic.plane_stride_bytes(0) < 0);
        for pl in 0..3 {
            let (pw, ph) = pic.plane_dimensions(pl).unwrap();
            let rows: Vec<_> = pic.plane_rows_u8(pl).collect();
            assert_eq!(rows.len(), ph);
            for (y, row) in rows.iter().enumerate() {
                assert_eq!(row.len(), pw);
                assert!(
                    row.iter().all(|&v| v == y as u8),
                    "plane {pl} row {y} came back out of order: {row:?}"
                );
            }
        }
    }
}
