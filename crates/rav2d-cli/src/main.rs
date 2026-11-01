mod info;
mod ivf;
mod md5;
mod sink;
mod source;

use std::fs::File;
use std::io::{self, BufWriter, IsTerminal, Write};
use std::process::ExitCode;
use std::time::Instant;

use clap::{ArgAction, Parser};
use rav2d::{Data, Decoder, Rav2dError, Settings};

use sink::{Format, Sink};
use source::{Kind, Source};

const AFTER_HELP: &str = "\
Examples:
  rav2d in.obu -o out.y4m              decode to a file
  rav2d in.ivf -o - | ffplay -         watch it
  rav2d in.obu --verify in.obu.md5     check against avmdec's hash
  rav2d in.obu --info                  what is in the stream

Exit status: 0 decoded (and the hash matched); 1 the stream failed to decode,
or the hash did not match; 2 bad arguments, unreadable input or unwritable
output.";

#[derive(Parser)]
#[command(name = "rav2d", about = "AV2 video decoder", version, after_help = AFTER_HELP)]
struct Args {
    /// Input: IVF, or a raw OBU stream (the AV2 test vector format); - reads stdin
    input: String,

    /// Output file; - is stdout. The format follows the extension (.y4m, .yuv, .md5)
    /// unless --muxer is given. Omit to decode without writing anything
    #[arg(short, long)]
    output: Option<String>,

    /// Output format, overriding the extension; y4m when writing to stdout
    #[arg(short = 'm', long, value_enum)]
    muxer: Option<Format>,

    /// Print the MD5 of the decoded planes, as `avmdec --md5` does
    #[arg(long, conflicts_with_all = ["output", "muxer"])]
    md5: bool,

    /// Compare the MD5 of the decoded planes with HASH, or with the hash in a file
    #[arg(long, value_name = "HASH|FILE")]
    verify: Option<String>,

    /// Print one line per frame and a summary of the stream instead of writing frames
    #[arg(long, conflicts_with_all = ["output", "muxer", "md5", "verify"])]
    info: bool,

    /// Stop after this many frames have been output
    #[arg(short = 'l', long, value_name = "N")]
    limit: Option<u64>,

    /// Decode but do not output the first N frames
    #[arg(long, value_name = "N", default_value_t = 0)]
    skip: u64,

    /// Number of threads (0 = auto); today only the output copy and film grain run in parallel
    #[arg(short = 't', long, default_value_t = 0)]
    threads: u32,

    /// Skip film grain synthesis
    #[arg(long)]
    no_grain: bool,

    /// Scalability operating point
    #[arg(long, value_name = "N", default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..32))]
    operating_point: u32,

    /// Output every spatial layer; false outputs only the highest one
    #[arg(long, value_name = "BOOL", default_value_t = true, action = ArgAction::Set)]
    all_layers: bool,

    /// Refuse frames larger than this many pixels (0 = no limit)
    #[arg(long, value_name = "PIXELS", default_value_t = 0)]
    size_limit: u32,

    /// Fail on streams that break the specification instead of decoding best-effort
    #[arg(long)]
    strict: bool,

    /// Keep decoding after a frame fails: the rest of that IVF frame is
    /// skipped (in a raw OBU stream, the rest of the stream). Exits with 1 anyway
    #[arg(long)]
    keep_going: bool,

    /// Use the Rust code instead of the NEON assembly (aarch64; x86 has no assembly)
    #[arg(long)]
    no_asm: bool,

    /// Print nothing but errors
    #[arg(short, long)]
    quiet: bool,
}

/// An error with the exit status it maps to.
pub struct CliError {
    code: u8,
    message: String,
}

impl CliError {
    /// The stream could not be decoded, or the output cannot represent it.
    pub fn decode(message: impl Into<String>) -> Self {
        Self {
            code: 1,
            message: message.into(),
        }
    }

    pub fn usage(message: impl Into<String>) -> Self {
        Self {
            code: 2,
            message: message.into(),
        }
    }

    pub fn output(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::BrokenPipe {
            // The reader went away (ffplay closed, `| head`): not an error.
            return Self {
                code: 0,
                message: String::new(),
            };
        }
        Self::usage(format!("cannot write the output: {e}"))
    }
}

/// The expected hash for --verify: the argument itself, or the first word of
/// the file it names (avmdec's .md5 files hold just the hash).
fn expected_md5(arg: &str) -> Result<String, CliError> {
    let is_hash = arg.len() == 32 && arg.bytes().all(|b| b.is_ascii_hexdigit());
    if is_hash {
        return Ok(arg.to_ascii_lowercase());
    }
    let text = std::fs::read_to_string(arg)
        .map_err(|e| CliError::usage(format!("--verify {arg}: not a hash, and {e}")))?;
    let word = text.split_whitespace().next().unwrap_or("");
    if word.len() == 32 && word.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(word.to_ascii_lowercase())
    } else {
        Err(CliError::usage(format!(
            "--verify {arg}: no MD5 in the file"
        )))
    }
}

fn open_output(path: &str) -> Result<Box<dyn Write>, CliError> {
    if path == "-" {
        return Ok(Box::new(BufWriter::new(io::stdout().lock())));
    }
    let file =
        File::create(path).map_err(|e| CliError::usage(format!("cannot create {path}: {e}")))?;
    Ok(Box::new(BufWriter::new(file)))
}

fn run(args: Args) -> Result<(), CliError> {
    let expected = args.verify.as_deref().map(expected_md5).transpose()?;

    // Pick the output before touching the input, so a bad -o fails fast.
    let output: Option<(Format, String)> = if args.md5 {
        Some((Format::Md5, "-".into()))
    } else if let Some(path) = &args.output {
        let format = match args.muxer {
            Some(f) => f,
            None if path == "-" => Format::Y4m,
            None => Format::from_path(path).ok_or_else(|| {
                CliError::usage(format!(
                    "cannot tell the format of {path} from its extension; use .y4m, .yuv, .md5 or --muxer"
                ))
            })?,
        };
        Some((format, path.clone()))
    } else if args.muxer.is_some() {
        return Err(CliError::usage("--muxer needs --output"));
    } else {
        None
    };
    let out_writer = match &output {
        Some((_, path)) => Some(open_output(path)?),
        None => None,
    };

    let (mut source, kind) = Source::open(&args.input)
        .map_err(|e| CliError::usage(format!("cannot read {}: {e}", args.input)))?;
    let log = !args.quiet;
    if log {
        match kind {
            Kind::Ivf {
                width,
                height,
                frames,
            } => eprintln!("rav2d: IVF {width}x{height}, {frames} frames"),
            Kind::Raw { bytes } => eprintln!("rav2d: raw OBU stream ({bytes} bytes)"),
        }
    }

    let mut sink = match (output, out_writer) {
        (Some((format, _)), Some(w)) => Some(Sink::new(format, w, source.fps())),
        _ => None,
    };
    // --verify hashes on its own, next to whatever is being written.
    let mut verify_md5 = expected.as_ref().map(|_| md5::Md5::new());
    let mut info = args.info.then(info::Info::default);
    let mut stdout = io::stdout().lock();

    if args.no_asm {
        // SAFETY: still single-threaded here, and the decoder reads the
        // variable once, after this, when it first picks a kernel.
        unsafe { std::env::set_var("RAV2D_NEON_OFF", "all") };
    }
    let settings = Settings {
        n_threads: args.threads,
        apply_grain: !args.no_grain,
        operating_point: args.operating_point,
        all_layers: args.all_layers,
        frame_size_limit: args.size_limit,
        strict_std_compliance: args.strict,
        ..Settings::default()
    };
    let mut decoder = Decoder::open(&settings)
        .map_err(|e| CliError::usage(format!("cannot open the decoder: {e}")))?;

    let progress = log && io::stderr().is_terminal() && !args.info;
    let start = Instant::now();
    let mut decoded = 0u64; // frames the decoder returned
    let mut written = 0u64; // frames past --skip
    let mut failed = 0u64;

    'stream: loop {
        if args.limit.is_some_and(|l| written >= l) {
            break;
        }
        let chunk = source
            .next_chunk()
            .map_err(|e| CliError::decode(format!("cannot read {}: {e}", args.input)))?;
        match chunk {
            Some(c) if c.is_empty() => continue,
            Some(c) => decoder.send_data(Some(Data::wrap(c))),
            None => decoder.send_data(None),
        }
        .map_err(|e| CliError::decode(format!("decoding failed: {e}")))?;

        loop {
            let pic = match decoder.get_picture() {
                Ok(pic) => pic,
                Err(Rav2dError::Again) => break,
                Err(Rav2dError::Eof) => break 'stream,
                Err(e) if args.keep_going => {
                    failed += 1;
                    if log {
                        eprintln!("rav2d: frame after #{decoded} failed: {e}");
                    }
                    continue;
                }
                Err(e) => {
                    return Err(CliError::decode(format!(
                        "decoding failed after {decoded} frames: {e}"
                    )));
                }
            };
            decoded += 1;
            if decoded <= args.skip {
                continue;
            }

            if let Some(md5) = &mut verify_md5 {
                sink::for_each_row(&pic, |row| {
                    md5.update(row);
                    Ok(())
                })
                .map_err(CliError::output)?;
            }
            if let Some(info) = &mut info {
                let line = info.frame(written, &pic);
                writeln!(stdout, "{line}").map_err(CliError::output)?;
            }
            if let Some(sink) = &mut sink {
                sink.write(&pic)?;
            }
            written += 1;

            if progress && written.is_multiple_of(100) {
                eprint!("\rrav2d: {written} frames");
            }
            if args.limit.is_some_and(|l| written >= l) {
                break 'stream;
            }
        }
    }

    if let Some(sink) = sink {
        sink.finish()?;
    }
    if let Some(info) = &info {
        writeln!(stdout, "{}", info.summary()).map_err(CliError::output)?;
    }
    stdout.flush().map_err(CliError::output)?;

    if log {
        let secs = start.elapsed().as_secs_f64();
        let skipped = if args.skip > 0 {
            format!(", {} skipped", decoded.min(args.skip))
        } else {
            String::new()
        };
        eprintln!(
            "{}rav2d: {written} frames in {secs:.2}s ({:.1} fps){skipped}",
            if progress { "\r" } else { "" },
            decoded as f64 / secs
        );
    }

    if let (Some(expected), Some(md5)) = (expected, verify_md5) {
        let got = md5.finish();
        if got != expected {
            return Err(CliError::decode(format!(
                "MD5 mismatch: expected {expected}, got {got}"
            )));
        }
        if log {
            eprintln!("rav2d: MD5 matches ({got})");
        }
    }
    if failed > 0 {
        return Err(CliError::decode(format!(
            "{failed} frames failed to decode"
        )));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if !e.message.is_empty() {
                eprintln!("rav2d: error: {}", e.message);
            }
            ExitCode::from(e.code)
        }
    }
}
