//! Compressed input: an IVF file read frame by frame, or a raw OBU stream (the
//! format of the AV2 test vectors) read whole. `-` reads standard input.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read};

use crate::ivf;

pub enum Source {
    Ivf {
        reader: Box<dyn BufRead>,
        fps: (u32, u32),
    },
    Raw {
        data: Option<Vec<u8>>,
    },
}

/// What the input turned out to be, for the one-line banner.
pub enum Kind {
    Ivf {
        width: u32,
        height: u32,
        frames: u32,
    },
    Raw {
        bytes: usize,
    },
}

impl Source {
    pub fn open(path: &str) -> io::Result<(Self, Kind)> {
        let mut reader: Box<dyn Read> = if path == "-" {
            Box::new(io::stdin().lock())
        } else {
            Box::new(File::open(path)?)
        };

        // Read the magic without assuming one read returns four bytes: a pipe
        // may hand them over in pieces.
        let mut head = Vec::with_capacity(4);
        (&mut reader).take(4).read_to_end(&mut head)?;
        let mut reader = BufReader::new(Cursor::new(head.clone()).chain(reader));

        if head == b"DKIF" {
            let hdr = ivf::read_header(&mut reader)?;
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
            let kind = Kind::Ivf {
                width: hdr.width,
                height: hdr.height,
                frames: hdr.num_frames,
            };
            Ok((
                Source::Ivf {
                    reader: Box::new(reader),
                    fps,
                },
                kind,
            ))
        } else {
            let mut data = Vec::new();
            reader.read_to_end(&mut data)?;
            let kind = Kind::Raw { bytes: data.len() };
            Ok((Source::Raw { data: Some(data) }, kind))
        }
    }

    /// Frame rate for the Y4M header. A raw OBU stream carries none that the
    /// decoder exposes, so it gets 30 fps.
    pub fn fps(&self) -> (u32, u32) {
        match self {
            Source::Ivf { fps, .. } => *fps,
            Source::Raw { .. } => (30, 1),
        }
    }

    /// Next chunk of compressed data, or `None` at end of stream.
    pub fn next_chunk(&mut self) -> io::Result<Option<Vec<u8>>> {
        match self {
            Source::Ivf { reader, .. } => Ok(ivf::read_frame(reader)?.map(|f| f.data)),
            Source::Raw { data } => Ok(data.take().filter(|d| !d.is_empty())),
        }
    }
}
