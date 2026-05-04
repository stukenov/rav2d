#[inline(always)]
pub fn imax(a: i32, b: i32) -> i32 {
    a.max(b)
}

#[inline(always)]
pub fn imin(a: i32, b: i32) -> i32 {
    a.min(b)
}

#[inline(always)]
pub fn umax(a: u32, b: u32) -> u32 {
    a.max(b)
}

#[inline(always)]
pub fn umin(a: u32, b: u32) -> u32 {
    a.min(b)
}

#[inline(always)]
pub fn iclip(v: i32, min: i32, max: i32) -> i32 {
    v.clamp(min, max)
}

#[inline(always)]
pub fn iclip64to32(v: i64, min: i32, max: i32) -> i32 {
    if v < min as i64 {
        min
    } else if v > max as i64 {
        max
    } else {
        v as i32
    }
}

#[inline(always)]
pub fn iclip_u8(v: i32) -> i32 {
    iclip(v, 0, 255)
}

#[inline(always)]
pub fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 { -v } else { v }
}

#[inline(always)]
pub fn apply_sign64(v: i64, s: i64) -> i32 {
    if s < 0 { -(v as i32) } else { v as i32 }
}

#[inline(always)]
pub fn clz(v: u32) -> u32 {
    v.leading_zeros()
}

#[inline(always)]
pub fn clzll(v: u64) -> u32 {
    v.leading_zeros()
}

#[inline(always)]
pub fn ulog2(v: u32) -> i32 {
    31 ^ clz(v) as i32
}

#[inline(always)]
pub fn u64log2(v: u64) -> i32 {
    63 ^ clzll(v) as i32
}

#[inline(always)]
pub fn ctz(v: u32) -> u32 {
    v.trailing_zeros()
}

#[inline(always)]
pub fn inv_recenter(r: u32, v: u32) -> u32 {
    if v > r << 1 {
        v
    } else if v & 1 == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

#[inline(always)]
pub fn popcnt(x: u32) -> u32 {
    x.count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulog2() {
        assert_eq!(ulog2(1), 0);
        assert_eq!(ulog2(2), 1);
        assert_eq!(ulog2(4), 2);
        assert_eq!(ulog2(255), 7);
        assert_eq!(ulog2(256), 8);
    }

    #[test]
    fn test_inv_recenter() {
        assert_eq!(inv_recenter(5, 0), 5);
        assert_eq!(inv_recenter(5, 1), 4);
        assert_eq!(inv_recenter(5, 2), 6);
        assert_eq!(inv_recenter(5, 11), 11);
    }

    #[test]
    fn test_iclip() {
        assert_eq!(iclip(-5, 0, 255), 0);
        assert_eq!(iclip(300, 0, 255), 255);
        assert_eq!(iclip(100, 0, 255), 100);
    }

    #[test]
    fn test_popcnt() {
        assert_eq!(popcnt(0), 0);
        assert_eq!(popcnt(1), 1);
        assert_eq!(popcnt(0xFF), 8);
        assert_eq!(popcnt(0xFFFFFFFF), 32);
    }
}
