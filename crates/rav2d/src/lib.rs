// clippy pedantic exception policy for rav2d (audit `-D clippy::pedantic`).
// rav2d is a faithful port of dav2d (C): the mechanical findings (~350 sites)
// were already fixed; the groups below are intentional port-of-dav2d idioms
// and remaining MaybeIncorrect-suggestion stragglers (Phase-2 candidates).
// Source attributes are used instead of cargo [lints] tables because cargo
// injects the latter as CLI --allow flags that a later `-D` would override.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::ptr_cast_constness)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::inline_always)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::stable_sort_primitive)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::decimal_bitwise_operands)]
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::single_match_else)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unnecessary_box_returns)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::format_push_string)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::same_functions_in_if_condition)]
#![allow(clippy::manual_slice_fill)]
#![allow(clippy::manual_is_variant_and)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::assert_is_empty)]
#![allow(clippy::manual_assert)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::format_collect)]

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
// The DSP/ported-C kernels carry explicit casts and parenthesised arithmetic
// that mirror the C source 1:1 for bit-exact auditability; these style lints
// would force divergence from the reference without changing behaviour.
#![allow(clippy::precedence)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::doc_lazy_continuation)]
#![warn(unsafe_op_in_unsafe_fn)]

/// Cache a rarely-set debug/trace environment flag so the hot decode path pays
/// the `getenv` cost — a process-global lock (`__findenv_locked`) plus a `String`
/// allocation — exactly once instead of on every block. The value is sampled on
/// first use, mirroring how `RAV2D_NEON_OFF` is handled in `simd.rs`. This keeps
/// the porting-era trace tooling available while removing it from the profile.
macro_rules! env_flag {
    ($name:literal) => {{
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var($name).is_ok())
    }};
}

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
pub(crate) mod ipred_neon;
pub(crate) mod ipred_prepare;
pub(crate) mod itx;
pub(crate) mod itx_1d;
pub(crate) mod lf_mask;
pub(crate) mod looprestoration;
pub(crate) mod mc;
pub(crate) mod mc_neon;
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
pub(crate) mod simd;
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
pub use picture::{
    EventFlags, PICTURE_ALIGNMENT, PicAllocator, Picture, PictureAllocation, PictureParameters,
};
