use crate::intops::clz;

pub static MSAC_RATE: [[u8; 3]; 125] = [
    [4,5,6],[4,5,5],[4,5,4],[4,5,7],[4,5,7],
    [4,4,6],[4,4,5],[4,4,4],[4,4,7],[4,4,7],
    [4,3,6],[4,3,5],[4,3,4],[4,3,7],[4,3,7],
    [4,6,6],[4,6,5],[4,6,4],[4,6,7],[4,6,7],
    [4,6,6],[4,6,5],[4,6,4],[4,6,7],[4,6,7],
    [3,5,6],[3,5,5],[3,5,4],[3,5,7],[3,5,7],
    [3,4,6],[3,4,5],[3,4,4],[3,4,7],[3,4,7],
    [3,3,6],[3,3,5],[3,3,4],[3,3,7],[3,3,7],
    [3,6,6],[3,6,5],[3,6,4],[3,6,7],[3,6,7],
    [3,6,6],[3,6,5],[3,6,4],[3,6,7],[3,6,7],
    [2,5,6],[2,5,5],[2,5,4],[2,5,7],[2,5,7],
    [2,4,6],[2,4,5],[2,4,4],[2,4,7],[2,4,7],
    [2,3,6],[2,3,5],[2,3,4],[2,3,7],[2,3,7],
    [2,6,6],[2,6,5],[2,6,4],[2,6,7],[2,6,7],
    [2,6,6],[2,6,5],[2,6,4],[2,6,7],[2,6,7],
    [5,5,6],[5,5,5],[5,5,4],[5,5,7],[5,5,7],
    [5,4,6],[5,4,5],[5,4,4],[5,4,7],[5,4,7],
    [5,3,6],[5,3,5],[5,3,4],[5,3,7],[5,3,7],
    [5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
    [5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
    [5,5,6],[5,5,5],[5,5,4],[5,5,7],[5,5,7],
    [5,4,6],[5,4,5],[5,4,4],[5,4,7],[5,4,7],
    [5,3,6],[5,3,5],[5,3,4],[5,3,7],[5,3,7],
    [5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
    [5,6,6],[5,6,5],[5,6,4],[5,6,7],[5,6,7],
];

#[repr(align(16))]
struct Aligned<T>(T);

static MSAC_MIN_PROB_INNER: Aligned<[[u16; 8]; 7]> = Aligned([
    [   63, 65535, 65535, 65535, 65535, 65535, 65535, 65535],
    [   47,    87, 65535, 65535, 65535, 65535, 65535, 65535],
    [   31,    63,    95, 65535, 65535, 65535, 65535, 65535],
    [   31,    55,    79,   103, 65535, 65535, 65535, 65535],
    [   23,    47,    63,    87,   111, 65535, 65535, 65535],
    [   23,    39,    55,    79,    95,   111, 65535, 65535],
    [   15,    31,    47,    63,    79,    95,   111, 65535],
]);

pub static MSAC_MIN_PROB: &[[u16; 8]; 7] = &MSAC_MIN_PROB_INNER.0;

pub struct MsacContext<'a> {
    buf_pos: usize,
    buf: &'a [u8],
    dif: u64,
    rng: u32,
    cnt: i32,
    allow_update_cdf: bool,
}

impl<'a> MsacContext<'a> {
    pub fn new(data: &'a [u8], disable_cdf_update_flag: bool) -> Self {
        let mut s = Self {
            buf_pos: 0,
            buf: data,
            dif: !0u64 >> 1,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: !disable_cdf_update_flag,
        };
        s.ctx_refill();
        s
    }

    #[inline]
    fn ctx_refill(&mut self) {
        let mut c = 40 - self.cnt;
        let mut dif = self.dif;
        loop {
            if self.buf_pos >= self.buf.len() {
                break;
            }
            dif ^= (self.buf[self.buf_pos] as u64) << c;
            self.buf_pos += 1;
            c -= 8;
            if c < 0 {
                break;
            }
        }
        self.dif = dif;
        self.cnt = 40 - c;
    }

    #[inline]
    fn ctx_norm(&mut self, dif: u64, rng: u32) {
        let d = 15 ^ (31 ^ clz(rng));
        let cnt = self.cnt;
        debug_assert!(rng <= 65535);
        self.dif = ((dif + 1) << d) - 1;
        self.rng = rng << d;
        self.cnt = cnt - d as i32;
        if (cnt as u32) < d {
            self.ctx_refill();
        }
    }

    pub fn decode_bools_bypass(&mut self, n_bits: u32) -> u32 {
        debug_assert!(n_bits > 0 && n_bits <= 32);
        if (self.cnt as u32) < n_bits {
            self.ctx_refill();
        }

        let r = self.rng as u64;
        let mut dif = self.dif;
        debug_assert!(r & 1 == 0);
        debug_assert!((dif >> 48) < r);
        let mut vw = r << 47;
        let mut ret: u32 = 0;
        for _ in 0..n_bits {
            ret <<= 1;
            if dif >= vw {
                dif -= vw;
            } else {
                ret |= 1;
            }
            vw >>= 1;
        }
        self.dif = ((dif + 1) << n_bits) - 1;
        self.cnt -= n_bits as i32;
        ret
    }

    #[inline]
    pub fn decode_bool_bypass(&mut self) -> u32 {
        self.decode_bools_bypass(1)
    }

    pub fn decode_unary_bypass(&mut self, max_bits: u32) -> u32 {
        debug_assert!(max_bits == 5 || max_bits == 6 || max_bits == 21);
        if (self.cnt as u32) < max_bits {
            self.ctx_refill();
        }

        let r = self.rng as u64;
        let mut dif = self.dif;
        debug_assert!(r & 1 == 0);
        debug_assert!((dif >> 48) < r);
        let mut vw = r << 47;
        let mut ret: u32 = 0;
        let mut bit: u32 = 0;
        while bit < max_bits {
            if dif >= vw {
                dif -= vw;
                vw >>= 1;
                ret += 1;
                bit += 1;
            } else {
                bit += 1;
                break;
            }
        }
        self.dif = ((dif + 1) << bit) - 1;
        self.cnt -= bit as i32;
        ret
    }

    fn decode_bool_raw(&mut self, f: u32) -> u32 {
        let r = self.rng;
        let dif = self.dif;
        debug_assert!((dif >> 48) < r as u64);
        let p = ((f >> 7) << 4) + 8;
        let mut v = ((r >> 8) * p >> 7) << 3;
        let vw = (v as u64) << 48;
        let ret = if dif >= vw { 1 } else { 0 };
        let new_dif = dif - ret as u64 * vw;
        if ret != 0 { v = r - v; }
        self.ctx_norm(new_dif, v);
        (ret == 0) as u32
    }

    pub fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        let c = (self.dif >> 48) as u32;
        let r = self.rng >> 8;
        let mut u: u32;
        let mut v = self.rng;
        let mut val: u32 = 0;
        let min_prob = &MSAC_MIN_PROB[n_symbols - 1];

        debug_assert!(n_symbols <= 7);

        loop {
            u = v;
            let p_raw = (cdf[val as usize] | 127) as i32 - min_prob[val as usize] as i32;
            let p = p_raw.max(0) as u32;
            v = (r * p >> 10) << 3;
            if c >= v {
                break;
            }
            val += 1;
        }

        debug_assert!(u <= self.rng);

        self.ctx_norm(self.dif - ((v as u64) << 48), u - v);

        if self.allow_update_cdf {
            let pc = cdf[n_symbols];
            let count = (pc & 0xFF) as u8;
            debug_assert!(count <= 32);
            let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize]
                + if n_symbols > 2 { 1 } else { 0 };
            for i in 0..val as usize {
                cdf[i] += ((32768 - cdf[i]) >> rate) as u16;
            }
            for i in val as usize..n_symbols {
                cdf[i] -= (cdf[i] >> rate) as u16;
            }
            cdf[n_symbols] = pc + if count < 32 { 1 } else { 0 };
        }

        val
    }

    pub fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        let bit = self.decode_bool_raw(cdf[0] as u32);

        if self.allow_update_cdf {
            let pc = cdf[1];
            let count = (pc & 0xFF) as u8;
            let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize];
            if bit != 0 {
                cdf[0] += ((32768 - cdf[0]) >> rate) as u16;
            } else {
                cdf[0] -= (cdf[0] >> rate) as u16;
            }
            cdf[1] = pc + if count < 32 { 1 } else { 0 };
        }

        bit
    }

    pub fn decode_uniform(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let l = crate::intops::ulog2(n) + 1;
        debug_assert!(l > 1);
        let m = (1u32 << l) - n;
        let v = self.decode_bools_bypass((l - 1) as u32);
        if v < m { v } else { (v << 1) - m + self.decode_bool_bypass() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msac_init() {
        let data = [0x00, 0x00, 0x00, 0x01];
        let s = MsacContext::new(&data, false);
        assert_eq!(s.rng, 0x8000);
        assert!(s.allow_update_cdf);
    }

    #[test]
    fn test_msac_rate_table_size() {
        assert_eq!(MSAC_RATE.len(), 125);
        assert_eq!(MSAC_RATE[0].len(), 3);
    }

    #[test]
    fn test_msac_min_prob_table_size() {
        assert_eq!(MSAC_MIN_PROB.len(), 7);
        assert_eq!(MSAC_MIN_PROB[0].len(), 8);
    }

    #[test]
    fn test_decode_bools_bypass() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut s = MsacContext::new(&data, true);
        let val = s.decode_bools_bypass(1);
        assert!(val <= 1);
    }
}
