pub static MAX_WIDTH_Y: [i8; 4] = [1, 3, 6, 8];
pub static MAX_WIDTH_UV: [i8; 3] = [1, 3, 4];

pub static Q_FIRST: [i8; 3] = [45, 40, 32];
pub static Q_THRESH_MULTS: [i8; 8] = [32, 25, 19, 19, 0, 18, 0, 17];
pub static W_MULT: [i8; 8] = [85, 51, 37, 28, 0, 20, 0, 15];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_width_y() {
        assert_eq!(MAX_WIDTH_Y[0], 1);
        assert_eq!(MAX_WIDTH_Y[3], 8);
    }

    #[test]
    fn test_max_width_uv() {
        assert_eq!(MAX_WIDTH_UV[0], 1);
        assert_eq!(MAX_WIDTH_UV[2], 4);
    }

    #[test]
    fn test_q_first() {
        assert!(Q_FIRST[0] > Q_FIRST[2]);
    }

    #[test]
    fn test_q_thresh_mults_zero_entries() {
        assert_eq!(Q_THRESH_MULTS[4], 0);
        assert_eq!(Q_THRESH_MULTS[6], 0);
    }

    #[test]
    fn test_w_mult_zero_entries() {
        assert_eq!(W_MULT[4], 0);
        assert_eq!(W_MULT[6], 0);
    }
}
