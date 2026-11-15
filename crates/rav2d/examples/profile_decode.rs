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

//! Loop-decode a raw OBU clip for profiling (`sample`/xctrace).
//! Usage: `profile_decode` <clip.obu> [iterations]
use std::path::PathBuf;

use rav2d::{Data, Decoder, Rav2dError, Settings};

fn decode_once(bytes: &[u8]) -> usize {
    let s = Settings {
        n_threads: 1,
        apply_grain: false,
        run_decode: true,
        ..Settings::default()
    };
    let mut dec = Decoder::open(&s).unwrap();
    let mut sent = false;
    let mut frames = 0usize;
    loop {
        if !sent {
            match dec.send_data(Some(Data::wrap(bytes.to_vec()))) {
                Ok(()) => sent = true,
                Err(Rav2dError::Again) => {}
                Err(_) => break,
            }
        }
        match dec.get_picture() {
            Ok(_) => frames += 1,
            Err(Rav2dError::Again) => {
                if sent {
                    let _ = dec.send_data(None);
                } else {
                    break;
                }
            }
            Err(_) => break,
        }
        if frames > 10_000 {
            break;
        }
    }
    frames
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(
        args.get(1)
            .expect("usage: profile_decode <clip.obu> [iters]"),
    );
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let bytes = std::fs::read(&path).expect("read clip");
    let mut total = 0usize;
    for _ in 0..iters {
        total = total.wrapping_add(decode_once(&bytes));
    }
    println!("decoded {total} frames over {iters} iters");
}
