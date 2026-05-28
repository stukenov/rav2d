use std::io::{self, Read};

const IVF_HEADER_SIZE: usize = 32;
const IVF_FRAME_HEADER_SIZE: usize = 12;

pub struct IvfHeader {
    pub width: u32,
    pub height: u32,
    pub timebase_num: u32,
    pub timebase_den: u32,
    pub num_frames: u32,
}

pub fn read_header<R: Read>(reader: &mut R) -> io::Result<IvfHeader> {
    let mut buf = [0u8; IVF_HEADER_SIZE];
    reader.read_exact(&mut buf)?;

    if &buf[0..4] != b"DKIF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not an IVF file"));
    }

    let codec = &buf[8..12];
    if codec != b"AV02" && codec != b"AV01" && codec != b"av02" && codec != b"av01" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported codec: {:?}", std::str::from_utf8(codec).unwrap_or("?")),
        ));
    }

    Ok(IvfHeader {
        width: u16::from_le_bytes([buf[12], buf[13]]) as u32,
        height: u16::from_le_bytes([buf[14], buf[15]]) as u32,
        timebase_num: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        timebase_den: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
        num_frames: u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]),
    })
}

pub struct IvfFrame {
    pub data: Vec<u8>,
    pub timestamp: u64,
}

pub fn read_frame<R: Read>(reader: &mut R) -> io::Result<Option<IvfFrame>> {
    let mut hdr = [0u8; IVF_FRAME_HEADER_SIZE];
    match reader.read_exact(&mut hdr) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let size = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
    let timestamp = u64::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7], hdr[8], hdr[9], hdr[10], hdr[11]]);

    let mut data = vec![0u8; size];
    reader.read_exact(&mut data)?;

    Ok(Some(IvfFrame { data, timestamp }))
}
