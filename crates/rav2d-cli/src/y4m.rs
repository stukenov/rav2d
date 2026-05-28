use std::io::{self, Write};

pub struct Y4mWriter<W: Write> {
    writer: W,
    header_written: bool,
}

impl<W: Write> Y4mWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            header_written: false,
        }
    }

    pub fn write_header(
        &mut self,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        colorspace: &str,
    ) -> io::Result<()> {
        writeln!(
            self.writer,
            "YUV4MPEG2 W{} H{} F{}:{} Ip C{}",
            width, height, fps_num, fps_den, colorspace
        )?;
        self.header_written = true;
        Ok(())
    }

    pub fn write_frame(&mut self, planes: &[&[u8]]) -> io::Result<()> {
        if !self.header_written {
            return Err(io::Error::other("header not written"));
        }
        self.writer.write_all(b"FRAME\n")?;
        for plane in planes {
            self.writer.write_all(plane)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
