//! MD5 over the decoded planes, so output can be checked against the hashes
//! `avmdec --md5` prints. Dependency-free on purpose: it is a few dozen lines,
//! and the same implementation is pinned by `md5_matches_known_vectors` in the
//! library's conformance tests.

pub struct Md5 {
    state: [u32; 4],
    buf: [u8; 64],
    buffered: usize,
    total: u64,
}

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

// floor(|sin(i + 1)| * 2^32), the standard constants.
const K: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

impl Md5 {
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: [0; 64],
            buffered: 0,
            total: 0,
        }
    }

    fn block(&mut self, chunk: &[u8]) {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            let sum = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(S[i]));
            a = tmp;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.buffered > 0 {
            let want = (64 - self.buffered).min(data.len());
            self.buf[self.buffered..self.buffered + want].copy_from_slice(&data[..want]);
            self.buffered += want;
            data = &data[want..];
            if self.buffered < 64 {
                return;
            }
            let chunk = self.buf;
            self.block(&chunk);
            self.buffered = 0;
        }
        while data.len() >= 64 {
            let (chunk, rest) = data.split_at(64);
            self.block(chunk);
            data = rest;
        }
        self.buf[..data.len()].copy_from_slice(data);
        self.buffered = data.len();
    }

    pub fn finish(mut self) -> String {
        let bits = self.total.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buffered != 56 {
            self.update(&[0x00]);
        }
        let mut chunk = self.buf;
        chunk[56..64].copy_from_slice(&bits.to_le_bytes());
        self.block(&chunk);
        self.state
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Md5;

    fn hash(data: &[u8]) -> String {
        let mut h = Md5::new();
        h.update(data);
        h.finish()
    }

    #[test]
    fn matches_known_vectors() {
        assert_eq!(hash(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hash(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            hash(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
        assert_eq!(hash(&[b'a'; 1000]), "cabe45dcc9ae5b66ba86600cca6b8ba8");
    }

    #[test]
    fn split_updates_match_one_update() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut h = Md5::new();
        for part in data.chunks(13) {
            h.update(part);
        }
        assert_eq!(h.finish(), hash(&data));
    }
}
