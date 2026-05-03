use crate::intops::apply_sign;

pub const PC_WIENER_NORMALIZER: [u16; 4] = [3739, 3273, 3074, 7];

static MODE_WEIGHTS: [[i16; 3]; 4] = [
    [-527, 15325, 321],
    [26436, -17705, 17905],
    [366, -147, -194],
    [202, -267, -179],
];

static MODE_OFFSETS: [i16; 4] = [-547, -21565, -573, -680];

pub fn get_qval_given_tskip(mut qstep: i32, tskip: i32, i: usize, bitdepth_min_8: i32) -> i32 {
    qstep = (qstep + ((1 << bitdepth_min_8) >> 1)) >> bitdepth_min_8;
    let prod = (tskip * qstep + 128) >> 8;
    let qval = MODE_WEIGHTS[i][0] as i32 * (tskip << 5)
        + MODE_WEIGHTS[i][1] as i32 * qstep
        + MODE_WEIGHTS[i][2] as i32 * prod;
    let abs_qval = qval.abs();
    let qval = apply_sign((abs_qval + (1 << 12)) >> 13, qval);
    255 * (MODE_OFFSETS[i] as i32 + qval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc_wiener_normalizer() {
        assert_eq!(PC_WIENER_NORMALIZER[3], 7);
    }

    #[test]
    fn test_get_qval_given_tskip_8bpc() {
        let v = get_qval_given_tskip(16, 100, 0, 0);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_get_qval_given_tskip_all_modes() {
        for i in 0..4 {
            let v = get_qval_given_tskip(32, 200, i, 0);
            let _ = v;
        }
    }

    #[test]
    fn test_get_qval_given_tskip_10bpc() {
        let v0 = get_qval_given_tskip(64, 300, 1, 0);
        let v2 = get_qval_given_tskip(64, 300, 1, 2);
        assert_ne!(v0, v2);
    }

    #[test]
    fn test_get_qval_given_tskip_zero() {
        let v = get_qval_given_tskip(0, 0, 0, 0);
        assert_eq!(v, 255 * MODE_OFFSETS[0] as i32);
    }
}
