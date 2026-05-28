mod ivf;
mod y4m;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::process;
use std::time::Instant;

use clap::Parser;
use rav2d::{Data, Decoder, Rav2dError, Settings};

#[derive(Parser)]
#[command(name = "rav2d", about = "AV2 video decoder", version)]
struct Args {
    /// Input file (IVF format)
    input: String,

    /// Output file (Y4M format). Omit for decode-only benchmark.
    #[arg(short, long)]
    output: Option<String>,

    /// Number of threads (0 = auto)
    #[arg(short = 't', long, default_value_t = 0)]
    threads: u32,

    /// Maximum number of frames to decode
    #[arg(short = 'l', long)]
    limit: Option<u64>,

    /// Skip film grain synthesis
    #[arg(long)]
    no_grain: bool,
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(&args.input)?;
    let mut reader = BufReader::new(file);

    let ivf_hdr = ivf::read_header(&mut reader)?;
    eprintln!(
        "rav2d: {}x{}, {} frames",
        ivf_hdr.width, ivf_hdr.height, ivf_hdr.num_frames
    );

    let mut settings = Settings::default();
    settings.n_threads = args.threads;
    settings.apply_grain = !args.no_grain;

    let mut decoder = Decoder::open(&settings)?;

    let mut y4m_out = args.output.as_ref().map(|path| {
        let file = File::create(path).expect("failed to create output file");
        y4m::Y4mWriter::new(BufWriter::new(file))
    });

    let mut frames_decoded = 0u64;
    let mut header_written = false;
    let start = Instant::now();

    loop {
        if args.limit.is_some_and(|l| frames_decoded >= l) {
            break;
        }

        match ivf::read_frame(&mut reader)? {
            Some(frame) => {
                decoder.send_data(Some(Data::wrap(frame.data)))?;
            }
            None => {
                decoder.send_data(None)?;
            }
        }

        loop {
            match decoder.get_picture() {
                Ok(pic) => {
                    frames_decoded += 1;

                    if let Some(ref mut writer) = y4m_out {
                        if !header_written {
                            let fps_num = if ivf_hdr.timebase_den > 0 { ivf_hdr.timebase_den } else { 30 };
                            let fps_den = if ivf_hdr.timebase_num > 0 { ivf_hdr.timebase_num } else { 1 };
                            writer.write_header(
                                pic.p.w as u32,
                                pic.p.h as u32,
                                fps_num,
                                fps_den,
                                "420",
                            )?;
                            header_written = true;
                        }

                        let y_size = pic.p.w as usize * pic.p.h as usize;
                        let uv_size = y_size / 4;
                        if let (Some(y_ptr), Some(u_ptr), Some(v_ptr)) =
                            (pic.data[0], pic.data[1], pic.data[2])
                        {
                            let y_plane = unsafe { std::slice::from_raw_parts(y_ptr.as_ptr(), y_size) };
                            let u_plane = unsafe { std::slice::from_raw_parts(u_ptr.as_ptr(), uv_size) };
                            let v_plane = unsafe { std::slice::from_raw_parts(v_ptr.as_ptr(), uv_size) };
                            writer.write_frame(&[y_plane, u_plane, v_plane])?;
                        }
                    }

                    if frames_decoded.is_multiple_of(100) {
                        eprint!("\rrav2d: {} frames decoded", frames_decoded);
                    }
                }
                Err(Rav2dError::Again) => break,
                Err(Rav2dError::Eof) => {
                    let elapsed = start.elapsed();
                    eprintln!(
                        "\rrav2d: {} frames decoded in {:.2}s ({:.1} fps)",
                        frames_decoded,
                        elapsed.as_secs_f64(),
                        frames_decoded as f64 / elapsed.as_secs_f64()
                    );
                    if let Some(ref mut writer) = y4m_out {
                        writer.flush()?;
                    }
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\rrav2d: {} frames decoded in {:.2}s ({:.1} fps)",
        frames_decoded,
        elapsed.as_secs_f64(),
        frames_decoded as f64 / elapsed.as_secs_f64()
    );
    if let Some(ref mut writer) = y4m_out {
        writer.flush()?;
    }

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("rav2d: error: {e}");
        process::exit(1);
    }
}
