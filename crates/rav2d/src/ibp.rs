use crate::intops::ulog2;
use crate::tables::DIV_RECIP;

#[inline]
fn fast_div32(num: u32, den: u32) -> u8 {
    let shift = ulog2(den) as u32;
    let rem = den - (1 << shift);
    let idx = ((rem << 7) + (1 << (shift - 1))) >> shift;
    debug_assert!(idx <= 128);
    let shift = shift + 2;
    let res = ((num as u64 * DIV_RECIP[idx as usize] as u64) + ((1u64 << shift) >> 1)) >> shift;
    debug_assert!(res < 256);
    res as u8
}

pub fn init_ibp_weights() -> [[[u8; 16]; 16]; 7] {
    const DR_DY_Q6: [u32; 7] = [682, 256, 170, 128, 81, 64, 50];
    let mut weights = [[[0u8; 16]; 16]; 7];
    for m in 0..7 {
        let dy = DR_DY_Q6[m];
        for y in 0..16 {
            let yy = ((y + 1) as u32) << 6;
            let mut y_pos = dy;
            for x in 0..16 {
                weights[m][y][x] = fast_div32(y_pos, yy + y_pos);
                y_pos += dy;
            }
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibp_weights_init() {
        let w = init_ibp_weights();
        assert_eq!(w.len(), 7);
        for m in 0..7 {
            for y in 0..16 {
                for x in 0..16 {
                    assert!(w[m][y][x] < 255, "weight overflow at [{m}][{y}][{x}]");
                }
                assert!(w[m][y][0] > 0, "first column should be nonzero at [{m}][{y}]");
            }
        }
    }

    #[test]
    fn test_ibp_weights_deterministic() {
        let w1 = init_ibp_weights();
        let w2 = init_ibp_weights();
        assert_eq!(w1, w2);
    }
}
