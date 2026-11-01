//! Where decoded frames go: Y4M, raw planar YUV, or an MD5 of the planes.

use std::io::{self, Write};

use rav2d::{Picture, PixelLayout};

use crate::CliError;
use crate::md5::Md5;

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Format {
    /// YUV4MPEG2, readable by ffmpeg, ffplay and mpv
    Y4m,
    /// Raw planar samples, no headers; 16-bit little-endian above 8 bits
    Yuv,
    /// MD5 of the raw planes, the same hash `avmdec --md5` prints
    Md5,
}

impl Format {
    /// The format an output path asks for by its extension.
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
        match ext.as_str() {
            "y4m" => Some(Format::Y4m),
            "yuv" => Some(Format::Yuv),
            "md5" => Some(Format::Md5),
            _ => None,
        }
    }
}

/// Feed every sample row of a picture, in plane order, to `f`. Rows carry no
/// padding, so the concatenation is exactly a `.yuv` frame.
pub fn for_each_row(pic: &Picture, mut f: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
    let planes = if pic.p.layout == PixelLayout::I400 {
        1
    } else {
        3
    };
    for plane in 0..planes {
        for row in pic.plane_rows_bytes(plane) {
            f(row)?;
        }
    }
    Ok(())
}

pub struct Sink {
    format: Format,
    out: Box<dyn Write>,
    fps: (u32, u32),
    /// Picture size the Y4M header was written with.
    y4m_size: Option<(i32, i32)>,
    md5: Md5,
}

impl Sink {
    pub fn new(format: Format, out: Box<dyn Write>, fps: (u32, u32)) -> Self {
        Self {
            format,
            out,
            fps,
            y4m_size: None,
            md5: Md5::new(),
        }
    }

    pub fn write(&mut self, pic: &Picture) -> Result<(), CliError> {
        match self.format {
            Format::Md5 => {
                let md5 = &mut self.md5;
                for_each_row(pic, |row| {
                    md5.update(row);
                    Ok(())
                })
                .map_err(CliError::output)
            }
            Format::Yuv => {
                for_each_row(pic, |row| self.out.write_all(row)).map_err(CliError::output)
            }
            Format::Y4m => {
                let size = (pic.p.w, pic.p.h);
                match self.y4m_size {
                    None => {
                        writeln!(
                            self.out,
                            "YUV4MPEG2 W{} H{} F{}:{} Ip C{}",
                            size.0,
                            size.1,
                            self.fps.0,
                            self.fps.1,
                            y4m_colorspace(pic.p.layout, pic.p.bpc)
                        )
                        .map_err(CliError::output)?;
                        self.y4m_size = Some(size);
                    }
                    Some(first) if first != size => {
                        return Err(CliError::decode(format!(
                            "the frame size changes from {}x{} to {}x{}, which Y4M cannot hold; \
                             write .yuv or .md5 instead",
                            first.0, first.1, size.0, size.1
                        )));
                    }
                    Some(_) => {}
                }
                self.out.write_all(b"FRAME\n").map_err(CliError::output)?;
                for_each_row(pic, |row| self.out.write_all(row)).map_err(CliError::output)
            }
        }
    }

    /// Flush the output; for MD5 that is when the hash gets written.
    pub fn finish(mut self) -> Result<(), CliError> {
        if self.format == Format::Md5 {
            let hash = std::mem::replace(&mut self.md5, Md5::new()).finish();
            writeln!(self.out, "{hash}").map_err(CliError::output)?;
        }
        self.out.flush().map_err(CliError::output)
    }
}

/// The Y4M colorspace tag for a picture's layout and bit depth.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_from_extension() {
        assert_eq!(Format::from_path("out.y4m"), Some(Format::Y4m));
        assert_eq!(Format::from_path("dir.v2/out.YUV"), Some(Format::Yuv));
        assert_eq!(Format::from_path("clip.obu.md5"), Some(Format::Md5));
        assert_eq!(Format::from_path("out.mp4"), None);
        assert_eq!(Format::from_path("out"), None);
    }

    #[test]
    fn y4m_colorspace_tags() {
        assert_eq!(y4m_colorspace(PixelLayout::I420, 8), "420jpeg");
        assert_eq!(y4m_colorspace(PixelLayout::I420, 10), "420p10");
        assert_eq!(y4m_colorspace(PixelLayout::I400, 10), "mono10");
        assert_eq!(y4m_colorspace(PixelLayout::I444, 12), "444p12");
    }
}
