use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once};

use crate::cpu;
use crate::data::Data;
use crate::dsp::{DSPContext, PalDSPContext, RefmvsDSPContext};
use crate::error::Rav2dError;
use crate::internal::DecoderContext;
use crate::log::Logger;
use crate::mem::MemPool;
use crate::obu;
use crate::picture::{DefaultPicAllocator, PicAllocator, Picture, ThreadPicture};

pub const MAX_THREADS: u32 = 256;
pub const MAX_FRAME_DELAY: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Which in-loop filters to apply during decoding.
#[non_exhaustive]
#[derive(Default)]
pub enum InloopFilterType {
    None = 0,
    Deblock = 1,
    Cdef = 2,
    Restoration = 4,
    Wiener = 8,
    Gdf = 16,
    #[default]
    All = 31,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Which frame types to decode.
#[non_exhaustive]
#[derive(Default)]
pub enum DecodeFrameType {
    #[default]
    All = 0,
    Reference = 1,
    Intra = 2,
    Key = 3,
}

/// Decoder configuration. Use `Settings::default()` for sensible defaults.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Number of worker threads. 0 = auto-detect from CPU count.
    pub n_threads: u32,
    /// Maximum frame delay for pipelining. 0 = auto based on thread count.
    pub max_frame_delay: u32,
    /// Apply film grain synthesis to decoded output.
    pub apply_grain: bool,
    /// Scalability operating point index (0–31).
    pub operating_point: u32,
    /// Output all temporal/spatial layers.
    pub all_layers: bool,
    /// Maximum frame size in pixels (width × height). 0 = unlimited.
    pub frame_size_limit: u32,
    /// Abort on spec-violating bitstreams instead of best-effort.
    pub strict_std_compliance: bool,
    /// Output frames not marked for display.
    pub output_invisible_frames: bool,
    /// Which in-loop filters to apply.
    pub inloop_filters: InloopFilterType,
    /// Which frame types to decode.
    pub decode_frame_type: DecodeFrameType,
    /// Bring-up gate: actually run reconstruction (intra only so far) and emit
    /// pictures. Default off while recon/filters are incomplete; enabled by the
    /// conformance harness. Will become unconditional once decode is complete.
    pub run_decode: bool,
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
            run_decode: false,
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

pub fn get_frame_delay(s: &Settings) -> Result<u32, Rav2dError> {
    if s.n_threads > MAX_THREADS || s.max_frame_delay > MAX_FRAME_DELAY {
        return Err(Rav2dError::InvalidParam);
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

/// AV2 bitstream decoder.
///
/// Feed compressed OBU data with [`send_data`](Self::send_data), then
/// pull decoded frames with [`get_picture`](Self::get_picture).
pub struct Decoder {
    logger: Logger,
    allocator: Arc<dyn PicAllocator>,
    inloop_filters: InloopFilterType,
    decode_frame_type: DecodeFrameType,

    n_tc: u32,
    n_fc: u32,

    ctx: DecoderContext,

    input: Data,
    drain: bool,
    flush: AtomicBool,

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
    /// Create a new decoder with the given settings.
    pub fn open(s: &Settings) -> Result<Self, Rav2dError> {
        init_internal();

        if s.n_threads > MAX_THREADS || s.max_frame_delay > MAX_FRAME_DELAY {
            return Err(Rav2dError::InvalidParam);
        }
        if s.operating_point > 31 {
            return Err(Rav2dError::InvalidParam);
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

        let ctx = DecoderContext {
            seq_hdr: None,
            frame_hdr: None,
            tile: Vec::new(),
            n_tile_data: 0,
            n_tiles: 0,
            refs: Default::default(),
            cdf: Vec::new(),
            dsp: Arc::new(std::array::from_fn(|_| DSPContext::default())),
            pal_dsp: PalDSPContext::default(),
            refmvs_dsp: RefmvsDSPContext::default(),
            content_light: None,
            mastering_display: None,
            ci: None,
            fgm: Default::default(),
            apply_grain: s.apply_grain,
            operating_point: s.operating_point as i32,
            operating_point_idc: 0,
            all_layers: s.all_layers,
            max_spatial_id: 0,
            frame_size_limit: s.frame_size_limit,
            strict_std_compliance: s.strict_std_compliance,
            output_invisible_frames: s.output_invisible_frames,
            n_passes: 1,
            run_decode: s.run_decode,
            frame_out: None,
        };

        Ok(Self {
            logger,
            allocator,
            inloop_filters: s.inloop_filters,
            decode_frame_type: s.decode_frame_type,
            n_tc,
            n_fc,
            ctx,
            input: Data::new(),
            drain: false,
            flush: AtomicBool::new(false),
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

    /// Feed compressed data to the decoder. Pass `None` to signal end-of-stream.
    ///
    /// Returns `Err(Again)` if the decoder hasn't consumed previous data yet;
    /// call `get_picture` to drain output before sending more.
    pub fn send_data(&mut self, data: Option<Data>) -> Result<(), Rav2dError> {
        match data {
            None => {
                self.drain = true;
                Ok(())
            }
            Some(d) => {
                if self.drain {
                    return Err(Rav2dError::Eof);
                }
                if d.is_empty() || d.len() > usize::MAX / 2 {
                    return Err(Rav2dError::InvalidParam);
                }
                if self.input.has_data() {
                    return Err(Rav2dError::Again);
                }
                self.input = d;
                Ok(())
            }
        }
    }

    /// Retrieve a decoded picture from the output queue.
    ///
    /// Returns `Err(Again)` when no picture is available yet (send more data).
    /// Returns `Err(Eof)` when the stream has been fully drained.
    pub fn get_picture(&mut self) -> Result<Picture, Rav2dError> {
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

    fn gen_picture(&mut self) -> Result<(), Rav2dError> {
        if self.output_picture_ready() {
            return Ok(());
        }

        while !self.input.is_empty() {
            let data = match self.input.data() {
                Some(d) => d,
                None => break,
            };
            match obu::parse_obus(&mut self.ctx, data) {
                Ok(consumed) => {
                    assert!(consumed <= self.input.len());
                    self.input.consume(consumed);
                    if self.input.is_empty() {
                        self.input.unref();
                    }
                    // A frame was reconstructed during parsing: enqueue it.
                    if let Some(pic) = self.ctx.frame_out.take() {
                        self.dpb[self.dpb_in].pic.p = pic;
                        self.dpb_in += 1;
                        if self.dpb_in == self.dpb_sz {
                            self.dpb_in = 0;
                        }
                    }
                }
                Err(_e) => {
                    self.input.unref();
                    return Err(Rav2dError::InvalidData);
                }
            }

            if self.output_picture_ready() {
                break;
            }
        }

        Ok(())
    }

    fn output_image(&mut self) -> Result<Picture, Rav2dError> {
        if self.dpb_in == self.dpb_out {
            if !self.drain {
                return Err(Rav2dError::Again);
            }
            self.drain = false;
            return Err(Rav2dError::Eof);
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

    /// Reset the decoder state, discarding all buffered data and references.
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

        for r in &mut self.ctx.refs {
            r.segmap = None;
            r.refmvs = None;
            r.ccsomap = None;
            r.p.frame_hdr = None;
            r.refpoc = [0; 7];
        }

        self.ctx.frame_hdr = None;
        self.ctx.seq_hdr = None;
        self.ctx.tile.clear();
        self.ctx.n_tile_data = 0;
        self.ctx.n_tiles = 0;

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
    1 << 8
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
        assert_eq!(get_frame_delay(&s), Err(Rav2dError::InvalidParam));
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
        assert_eq!(d.send_data(Some(data)), Err(Rav2dError::Eof));
    }

    #[test]
    fn test_send_data_empty() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        let data = Data::new();
        assert_eq!(d.send_data(Some(data)), Err(Rav2dError::InvalidParam));
    }

    #[test]
    fn test_send_data_double() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        d.send_data(Some(Data::wrap(vec![1, 2, 3]))).unwrap();
        assert_eq!(
            d.send_data(Some(Data::wrap(vec![4, 5, 6]))),
            Err(Rav2dError::Again)
        );
    }

    #[test]
    fn test_get_picture_no_data() {
        let s = Settings::default();
        let mut d = Decoder::open(&s).unwrap();
        assert_eq!(d.get_picture().err(), Some(Rav2dError::Again));
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
