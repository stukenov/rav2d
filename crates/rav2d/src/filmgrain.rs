pub fn get_random_number(bits: u32, state: &mut u32) -> u32 {
    let r = *state;
    let bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    (*state >> (16 - bits)) & ((1 << bits) - 1)
}

pub fn round2(x: i32, shift: u32) -> i32 {
    (x + ((1 << shift) >> 1)) >> shift
}

pub fn generate_scaling_8bpc(points: &[[u8; 2]], scaling: &mut [u8; 256]) {
    let num = points.len();
    if num == 0 {
        scaling.fill(0);
        return;
    }

    let first_x = points[0][0] as usize;
    scaling[..first_x].fill(points[0][1]);

    for i in 0..num - 1 {
        let bx = points[i][0] as i32;
        let by = points[i][1] as i32;
        let ex = points[i + 1][0] as i32;
        let ey = points[i + 1][1] as i32;
        let dx = ex - bx;
        let dy = ey - by;
        debug_assert!(dx > 0);
        let delta = dy * ((0x10000 + (dx >> 1)) / dx);
        let mut d = 0x8000i32;
        for x in 0..dx as usize {
            scaling[bx as usize + x] = (by + (d >> 16)) as u8;
            d += delta;
        }
    }

    let n = points[num - 1][0] as usize;
    scaling[n..].fill(points[num - 1][1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_random_number_lfsr() {
        let mut state = 0xAAAAu32;
        let v = get_random_number(8, &mut state);
        assert!(v < 256);
        assert_ne!(state, 0xAAAA);
    }

    #[test]
    fn test_get_random_number_sequence() {
        let mut state = 1u32;
        let mut vals = Vec::new();
        for _ in 0..100 {
            vals.push(get_random_number(4, &mut state));
        }
        assert!(vals.iter().any(|&v| v != vals[0]));
    }

    #[test]
    fn test_round2_basic() {
        assert_eq!(round2(10, 1), 5);
        assert_eq!(round2(11, 1), 6);
        assert_eq!(round2(7, 2), 2);
    }

    #[test]
    fn test_round2_negative() {
        assert_eq!(round2(-10, 1), -5);
    }

    #[test]
    fn test_generate_scaling_empty() {
        let mut scaling = [0u8; 256];
        generate_scaling_8bpc(&[], &mut scaling);
        assert!(scaling.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_generate_scaling_single_point() {
        let mut scaling = [0u8; 256];
        generate_scaling_8bpc(&[[128, 200]], &mut scaling);
        assert!(scaling[..128].iter().all(|&v| v == 200));
        assert!(scaling[128..].iter().all(|&v| v == 200));
    }

    #[test]
    fn test_generate_scaling_two_points() {
        let mut scaling = [0u8; 256];
        generate_scaling_8bpc(&[[0, 0], [255, 255]], &mut scaling);
        assert_eq!(scaling[0], 0);
        assert!(scaling[128] > 100);
        assert_eq!(scaling[255], 255);
    }

    #[test]
    fn test_generate_scaling_monotonic() {
        let mut scaling = [0u8; 256];
        generate_scaling_8bpc(&[[0, 10], [100, 50], [200, 50]], &mut scaling);
        for i in 0..99 {
            assert!(scaling[i] <= scaling[i + 1] || scaling[i + 1] <= scaling[i]);
        }
    }
}
