//! `--info`: one line per output frame, then what the stream is.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use rav2d::{Picture, PixelLayout};

fn layout_name(layout: PixelLayout) -> &'static str {
    match layout {
        PixelLayout::I400 => "4:0:0",
        PixelLayout::I420 => "4:2:0",
        PixelLayout::I422 => "4:2:2",
        PixelLayout::I444 => "4:4:4",
    }
}

#[derive(Default)]
pub struct Info {
    frames: u64,
    frame_types: BTreeMap<String, u64>,
    sizes: Vec<(i32, i32)>,
    grain_frames: u64,
    summary: Option<String>,
}

impl Info {
    /// Record a frame and return its line.
    pub fn frame(&mut self, index: u64, pic: &Picture) -> String {
        let (w, h) = (pic.p.w, pic.p.h);
        if !self.sizes.contains(&(w, h)) {
            self.sizes.push((w, h));
        }
        let mut line = format!(
            "frame {index}: {w}x{h} {}-bit {}",
            pic.p.bpc,
            layout_name(pic.p.layout)
        );
        if let Some(fh) = &pic.frame_hdr {
            let ty = format!("{:?}", fh.frame_type).to_lowercase();
            let _ = write!(line, " {ty}");
            if fh.show_existing_frame != 0 {
                line.push_str(" (shown again)");
            }
            if fh.tlayer_id != 0 || fh.mlayer_id != 0 {
                let _ = write!(line, " tlayer {} mlayer {}", fh.tlayer_id, fh.mlayer_id);
            }
            *self.frame_types.entry(ty).or_default() += 1;
        }
        if pic.fgm.is_some() {
            line.push_str(" grain");
            self.grain_frames += 1;
        }
        if self.summary.is_none()
            && let Some(sh) = &pic.seq_hdr
        {
            let mut s = format!(
                "profile {}, level {}, tier {}, {}-bit {}, up to {}x{}",
                sh.profile,
                sh.level,
                sh.tier,
                8 + 2 * sh.hbd as u32,
                layout_name(sh.layout),
                sh.max_width,
                sh.max_height
            );
            if sh.still_picture {
                s.push_str(", still picture");
            }
            if sh.max_tlayer_id > 0 || sh.max_mlayer_id > 0 {
                let _ = write!(
                    s,
                    ", {} temporal and {} spatial layers",
                    sh.max_tlayer_id + 1,
                    sh.max_mlayer_id + 1
                );
            }
            if sh.film_grain_present {
                s.push_str(", film grain");
            }
            self.summary = Some(s);
        }
        self.frames += 1;
        line
    }

    pub fn summary(&self) -> String {
        let mut s = format!(
            "stream: {}\n",
            self.summary.as_deref().unwrap_or("no sequence header")
        );
        let types: Vec<String> = self
            .frame_types
            .iter()
            .map(|(t, n)| format!("{n} {t}"))
            .collect();
        let _ = write!(s, "frames: {}", self.frames);
        if !types.is_empty() {
            let _ = write!(s, " ({})", types.join(", "));
        }
        if self.grain_frames > 0 {
            let _ = write!(s, ", {} with film grain", self.grain_frames);
        }
        if self.sizes.len() > 1 {
            let sizes: Vec<String> = self.sizes.iter().map(|(w, h)| format!("{w}x{h}")).collect();
            let _ = write!(s, "\nsizes: {}", sizes.join(", "));
        }
        s
    }
}
