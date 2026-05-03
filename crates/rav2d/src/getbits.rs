use crate::intops::{inv_recenter, ulog2};

pub struct GetBits<'a> {
    state: u64,
    bits_left: i32,
    error: bool,
    ptr: usize,
    data: &'a [u8],
}

impl<'a> GetBits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        assert!(!data.is_empty());
        Self {
            state: 0,
            bits_left: 0,
            error: false,
            ptr: 0,
            data,
        }
    }

    #[inline]
    pub fn has_error(&self) -> bool {
        self.error
    }

    #[inline]
    pub fn pos(&self) -> u32 {
        (self.ptr as u32) * 8 - self.bits_left as u32
    }

    pub fn get_bit(&mut self) -> u32 {
        if self.bits_left == 0 {
            if self.ptr >= self.data.len() {
                self.error = true;
                return 0;
            }
            let byte = self.data[self.ptr] as u64;
            self.ptr += 1;
            self.bits_left = 7;
            self.state = byte << 57;
            return (byte >> 7) as u32;
        }

        let state = self.state;
        self.bits_left -= 1;
        self.state = state << 1;
        (state >> 63) as u32
    }

    #[inline]
    fn refill(&mut self, n: i32) {
        debug_assert!(self.bits_left >= 0 && self.bits_left < 32);
        let mut st: u32 = 0;
        loop {
            if self.ptr >= self.data.len() {
                self.error = true;
                if st != 0 {
                    break;
                }
                return;
            }
            st = (st << 8) | self.data[self.ptr] as u32;
            self.ptr += 1;
            self.bits_left += 8;
            if n <= self.bits_left {
                break;
            }
        }
        self.state |= (st as u64) << (64 - self.bits_left);
    }

    pub fn get_bits(&mut self, n: i32) -> u32 {
        debug_assert!(n > 0 && n <= 32);
        if n as u32 > self.bits_left as u32 {
            self.refill(n);
        }
        let state = self.state;
        self.bits_left -= n;
        self.state = state << n;
        (state >> (64 - n)) as u32
    }

    pub fn get_sbits(&mut self, n: i32) -> i32 {
        debug_assert!(n > 0 && n <= 32);
        if n as u32 > self.bits_left as u32 {
            self.refill(n);
        }
        let state = self.state;
        self.bits_left -= n;
        self.state = state << n;
        ((state as i64) >> (64 - n)) as i32
    }

    pub fn get_uleb128(&mut self) -> u32 {
        let mut val: u64 = 0;
        let mut i: u32 = 0;

        loop {
            let v = self.get_bits(8);
            let more = v & 0x80;
            val |= ((v & 0x7F) as u64) << i;
            i += 7;
            if more == 0 || i >= 56 {
                break;
            }
        }

        if val > u32::MAX as u64 {
            self.error = true;
            return 0;
        }

        val as u32
    }

    pub fn get_golomb(&mut self, k: u32) -> u32 {
        debug_assert!(k < 32);
        let mut bits: u32 = 0;
        while bits < 32 - k {
            if self.get_bit() == 0 {
                break;
            }
            bits += 1;
        }
        if bits + k == 32 {
            return u32::MAX;
        }
        (bits << k) | self.get_bits(k as i32)
    }

    pub fn get_uniform(&mut self, max: u32) -> u32 {
        debug_assert!(max > 1);
        let l = ulog2(max) + 1;
        debug_assert!(l > 1);
        let m = (1u32 << l) - max;
        let v = self.get_bits(l - 1);
        if v < m { v } else { (v << 1) - m + self.get_bit() }
    }

    pub fn get_vlc(&mut self) -> u32 {
        if self.get_bit() != 0 {
            return 0;
        }

        let mut n_bits: i32 = 0;
        loop {
            n_bits += 1;
            if n_bits == 32 {
                return u32::MAX;
            }
            if self.get_bit() != 0 {
                break;
            }
        }

        ((1u32 << n_bits) - 1) + self.get_bits(n_bits)
    }

    pub fn get_bits_subexp_u(&mut self, ref_val: u32, n: u32, k: i32) -> u32 {
        let mut v: u32 = 0;

        let mut i = 0;
        loop {
            let b = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b;

            if n <= v + 3 * a {
                v += self.get_uniform(n - v);
                break;
            }

            if self.get_bit() == 0 {
                v += self.get_bits(b);
                break;
            }

            v += a;
            i += 1;
        }

        if ref_val * 2 <= n {
            inv_recenter(ref_val, v)
        } else {
            n - 1 - inv_recenter(n - 1 - ref_val, v)
        }
    }

    pub fn get_bits_subexp(&mut self, ref_val: i32, n: u32) -> i32 {
        let off = n as i32 - 1;
        let n2 = n + off as u32;
        self.get_bits_subexp_u((ref_val + off) as u32, n2, 3) as i32 - off
    }

    pub fn get_ref_uniform(&mut self, max: u32, def: u32) -> u32 {
        if self.get_bit() == 0 {
            return def;
        }
        let res = self.get_uniform(max - 1);
        res + if res >= def { 1 } else { 0 }
    }

    #[inline]
    pub fn bytealign(&mut self) {
        debug_assert!(self.bits_left <= 7);
        self.bits_left = 0;
        self.state = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bits_basic() {
        let data = [0b10110100, 0b11000000];
        let mut gb = GetBits::new(&data);

        assert_eq!(gb.get_bit(), 1);
        assert_eq!(gb.get_bit(), 0);
        assert_eq!(gb.get_bits(3), 0b110);
        assert_eq!(gb.get_bits(3), 0b100);
        assert!(!gb.has_error());
    }

    #[test]
    fn test_get_bits_position() {
        let data = [0xFF; 4];
        let mut gb = GetBits::new(&data);

        assert_eq!(gb.pos(), 0);
        gb.get_bits(8);
        assert_eq!(gb.pos(), 8);
        gb.get_bits(4);
        assert_eq!(gb.pos(), 12);
    }

    #[test]
    fn test_get_sbits() {
        let data = [0xFF, 0x00];
        let mut gb = GetBits::new(&data);

        let val = gb.get_sbits(8);
        assert_eq!(val, -1);
    }

    #[test]
    fn test_error_on_overread() {
        let data = [0xFF];
        let mut gb = GetBits::new(&data);

        gb.get_bits(8);
        gb.get_bit();
        assert!(gb.has_error());
    }

    #[test]
    fn test_uleb128() {
        let data = [0x80 | 0x01, 0x02];
        let mut gb = GetBits::new(&data);
        let val = gb.get_uleb128();
        assert_eq!(val, 0x01 | (0x02 << 7));
        assert!(!gb.has_error());
    }

    #[test]
    fn test_bytealign() {
        let data = [0xFF, 0xAA];
        let mut gb = GetBits::new(&data);
        gb.get_bits(3);
        gb.bytealign();
        assert_eq!(gb.bits_left, 0);
    }
}
