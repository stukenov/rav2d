//! Loop-decode a raw OBU clip for profiling (`sample`/xctrace).
//! Usage: profile_decode <clip.obu> [iterations]
use std::path::PathBuf;

use rav2d::{Data, Decoder, Rav2dError, Settings};

fn decode_once(bytes: &[u8]) -> usize {
    let mut s = Settings::default();
    s.n_threads = 1;
    s.apply_grain = false;
    s.run_decode = true;
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
    let path = PathBuf::from(args.get(1).expect("usage: profile_decode <clip.obu> [iters]"));
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let bytes = std::fs::read(&path).expect("read clip");
    let mut total = 0usize;
    for _ in 0..iters {
        total = total.wrapping_add(decode_once(&bytes));
    }
    println!("decoded {total} frames over {iters} iters");
}
