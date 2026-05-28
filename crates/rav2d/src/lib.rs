//! # rav2d — AV2 Video Decoder in Rust
//!
//! A safe Rust port of [dav2d](https://code.videolan.org/videolan/dav2d), the AV2 reference decoder.
//! Assembly-optimized DSP kernels are shared via FFI; all C parsing, decode orchestration,
//! and filter logic has been rewritten in safe Rust.
//!
//! ## Quick Start
//!
//! ```no_run
//! use rav2d::{Decoder, Settings, Data, Rav2dError};
//!
//! let mut decoder = Decoder::open(&Settings::default()).unwrap();
//!
//! // Feed compressed data
//! let obu_data: Vec<u8> = std::fs::read("input.obu").unwrap();
//! decoder.send_data(Some(Data::wrap(obu_data))).unwrap();
//!
//! // Retrieve decoded pictures
//! loop {
//!     match decoder.get_picture() {
//!         Ok(picture) => { /* process decoded frame */ }
//!         Err(Rav2dError::Again) => break, // need more input
//!         Err(Rav2dError::Eof) => break,   // end of stream
//!         Err(e) => panic!("decode error: {e}"),
//!     }
//! }
//! ```

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::while_immutable_condition)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(clippy::enum_variant_names)]
#![warn(unsafe_op_in_unsafe_fn)]

pub(crate) mod ccso;
pub(crate) mod cdef;
pub(crate) mod cdf;
pub(crate) mod cpu;
pub(crate) mod ctx;
pub(crate) mod deblock;
pub(crate) mod decode;
pub(crate) mod dip_tables;
pub(crate) mod dsp;
pub(crate) mod env;
pub(crate) mod filmgrain;
pub(crate) mod gdf_tables;
pub(crate) mod getbits;
pub(crate) mod ibp;
pub(crate) mod internal;
pub(crate) mod intops;
pub(crate) mod intra_edge;
pub(crate) mod ipred;
pub(crate) mod ipred_prepare;
pub(crate) mod itx;
pub(crate) mod itx_1d;
pub(crate) mod lf_mask;
pub(crate) mod looprestoration;
pub(crate) mod mc;
pub(crate) mod mem;
pub(crate) mod msac;
pub(crate) mod obu;
pub(crate) mod pal;
pub(crate) mod pixel;
pub(crate) mod quantizer;
pub(crate) mod recon;
pub(crate) mod ref_count;
pub(crate) mod refmvs;
pub(crate) mod scan;
pub(crate) mod stx;
pub(crate) mod stx_tables;
pub(crate) mod tables;
pub(crate) mod thread_task;
pub(crate) mod warpmv;
pub(crate) mod wedge;

mod data;
mod decoder;
mod error;
mod headers;
mod levels;
mod log;
mod picture;

pub use data::Data;
pub use decoder::{
    DecodeFrameType, Decoder, InloopFilterType, MAX_FRAME_DELAY, MAX_THREADS, Settings,
};
pub use decoder::{get_frame_delay, version, version_api};
pub use error::Rav2dError;
pub use headers::{FrameHeader, PixelLayout, SequenceHeader};
pub use log::Logger;
pub use picture::{EventFlags, PicAllocator, Picture};

pub use rav2d_sys as sys;
