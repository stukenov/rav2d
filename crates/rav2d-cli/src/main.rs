mod ivf;
mod y4m;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::process;
use std::time::Instant;

use clap::Parser;
use rav2d::{Data, Decoder, Picture, PixelLayout, Rav2dError, Settings};

#[derive(Parser)]
#[command(name = "rav2d", about = "AV2 video decoder", version)]
struct Args {
    /// Input file (IVF or raw OBU stream)
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

/// Demuxed input: either IVF frames read on demand, or the whole file as one
/// raw OBU stream (the format of AV2 conformance/test vectors).
enum Source {
    Ivf {
        reader: BufReader<File>,
        fps: (u32, u32),
    },
    Raw {
        data: Option<Vec<u8>>,
    },
}

impl Source {
    fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        let got = file.read(&mut magic)?;
        if got == 4 && &magic == b"DKIF" {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            let hdr = ivf::read_header(&mut reader)?;
            eprintln!(
                "rav2d: IVF {}x{}, {} frames",
                hdr.width, hdr.height, hdr.num_frames
            );
            let fps = (
                if hdr.timebase_den > 0 {
                    hdr.timebase_den
                } else {
                    30
                },
                if hdr.timebase_num > 0 {
                    hdr.timebase_num
                } else {
                    1
                },
            );
            Ok(Source::Ivf { reader, fps })
        } else {
            let mut data = Vec::new();
            File::open(path)?.read_to_end(&mut data)?;
            eprintln!("rav2d: raw OBU stream ({} bytes)", data.len());
            Ok(Source::Raw { data: Some(data) })
        }
    }

    fn fps(&self) -> (u32, u32) {
        match self {
            Source::Ivf { fps, .. } => *fps,
            Source::Raw { .. } => (30, 1),
        }
    }

    /// Next chunk of compressed data, or `None` at end of stream.
    fn next(&mut self) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        match self {
            Source::Ivf { reader, .. } => Ok(ivf::read_frame(reader)?.map(|f| f.data)),
            Source::Raw { data } => Ok(data.take()),
        }
    }
}

/// The y4m colorspace tag for a picture's layout and bit depth.
fn y4m_colorspace(layout: PixelLayout, bpc: i32) -> String {
    let base = match layout {
        PixelLayout::I400 => "mono",
        PixelLayout::I420 => "420jpeg",
        PixelLayout::I422 => "422",
        PixelLayout::I444 => "444",
    };
    match (layout, bpc) {
        (_, 8) => base.to_string(),
        (PixelLayout::I420, _) => format!("420p{bpc}"),
        (PixelLayout::I400, _) => format!("mono{bpc}"),
        _ => format!("{base}p{bpc}"),
    }
}

/// Copy one plane into a tightly-packed buffer (`w` samples per row), honouring
/// the picture's byte stride and bytes-per-sample.
fn pack_plane(pic: &Picture, plane: usize, w: usize, h: usize, bps: usize) -> Vec<u8> {
    let stride = pic.stride[if plane == 0 { 0 } else { 1 }].unsigned_abs();
    let row_bytes = w * bps;
    let mut out = vec![0u8; row_bytes * h];
    if let Some(ptr) = pic.data[plane] {
        // SAFETY: the allocation spans at least `stride * h` bytes per plane
        // (DefaultPicAllocator); rows are copied within that span.
        let src = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), stride * h) };
        for y in 0..h {
            out[y * row_bytes..(y + 1) * row_bytes]
                .copy_from_slice(&src[y * stride..y * stride + row_bytes]);
        }
    }
    out
}

fn write_picture<W: std::io::Write>(
    writer: &mut y4m::Y4mWriter<W>,
    pic: &Picture,
) -> std::io::Result<()> {
    let w = pic.p.w as usize;
    let h = pic.p.h as usize;
    let bps = if pic.p.bpc > 8 { 2 } else { 1 };
    let layout = pic.p.layout;
    let y = pack_plane(pic, 0, w, h, bps);
    if layout == PixelLayout::I400 {
        return writer.write_frame(&[&y]);
    }
    let ss_hor = (layout != PixelLayout::I444) as usize;
    let ss_ver = (layout == PixelLayout::I420) as usize;
    let cw = (w + ss_hor) >> ss_hor;
    let ch = (h + ss_ver) >> ss_ver;
    let u = pack_plane(pic, 1, cw, ch, bps);
    let v = pack_plane(pic, 2, cw, ch, bps);
    writer.write_frame(&[&y, &u, &v])
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut source = Source::open(&args.input)?;

    let settings = Settings {
        n_threads: args.threads,
        apply_grain: !args.no_grain,
        ..Settings::default()
    };

    let mut decoder = Decoder::open(&settings)?;

    let mut y4m_out = args.output.as_ref().map(|path| {
        let file = File::create(path).expect("failed to create output file");
        y4m::Y4mWriter::new(BufWriter::new(file))
    });

    let mut frames_decoded = 0u64;
    let mut header_written = false;
    let start = Instant::now();

    let finish = |frames_decoded: u64,
                  start: Instant,
                  y4m_out: &mut Option<y4m::Y4mWriter<BufWriter<File>>>|
     -> Result<(), Box<dyn std::error::Error>> {
        let elapsed = start.elapsed();
        eprintln!(
            "\rrav2d: {} frames decoded in {:.2}s ({:.1} fps)",
            frames_decoded,
            elapsed.as_secs_f64(),
            frames_decoded as f64 / elapsed.as_secs_f64()
        );
        if let Some(writer) = y4m_out {
            writer.flush()?;
        }
        Ok(())
    };

    loop {
        if args.limit.is_some_and(|l| frames_decoded >= l) {
            break;
        }

        match source.next()? {
            Some(chunk) => decoder.send_data(Some(Data::wrap(chunk)))?,
            None => decoder.send_data(None)?,
        }

        loop {
            match decoder.get_picture() {
                Ok(pic) => {
                    frames_decoded += 1;

                    if let Some(ref mut writer) = y4m_out {
                        if !header_written {
                            let (fps_num, fps_den) = source.fps();
                            writer.write_header(
                                pic.p.w as u32,
                                pic.p.h as u32,
                                fps_num,
                                fps_den,
                                &y4m_colorspace(pic.p.layout, pic.p.bpc),
                            )?;
                            header_written = true;
                        }
                        write_picture(writer, &pic)?;
                    }

                    if frames_decoded.is_multiple_of(100) {
                        eprint!("\rrav2d: {} frames decoded", frames_decoded);
                    }
                }
                Err(Rav2dError::Again) => break,
                Err(Rav2dError::Eof) => {
                    return finish(frames_decoded, start, &mut y4m_out);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    finish(frames_decoded, start, &mut y4m_out)
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("rav2d: error: {e}");
        process::exit(1);
    }
}
