pub static CCSO_POS: [[i8; 2]; 7] = [
    [-1, 0],
    [0, -1],
    [-1, -1],
    [-1, 1],
    [-1, -2],
    [1, -2],
    [0, 2],
];

#[inline(always)]
pub fn ccso_score(diff: i32, quant_step: i32, edge_classifier: u32) -> u32 {
    if diff > quant_step && edge_classifier == 0 {
        return 2;
    }
    if diff < -quant_step {
        return 0;
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccso_pos_table() {
        assert_eq!(CCSO_POS[0], [-1, 0]);
        assert_eq!(CCSO_POS[6], [0, 2]);
        assert_eq!(CCSO_POS.len(), 7);
    }

    #[test]
    fn test_ccso_score_large_positive() {
        assert_eq!(ccso_score(20, 10, 0), 2);
    }

    #[test]
    fn test_ccso_score_large_positive_with_edge() {
        assert_eq!(ccso_score(20, 10, 1), 1);
    }

    #[test]
    fn test_ccso_score_large_negative() {
        assert_eq!(ccso_score(-20, 10, 0), 0);
    }

    #[test]
    fn test_ccso_score_within_range() {
        assert_eq!(ccso_score(5, 10, 0), 1);
        assert_eq!(ccso_score(-5, 10, 0), 1);
    }

    #[test]
    fn test_ccso_score_boundary() {
        assert_eq!(ccso_score(10, 10, 0), 1);
        assert_eq!(ccso_score(-10, 10, 0), 1);
        assert_eq!(ccso_score(11, 10, 0), 2);
        assert_eq!(ccso_score(-11, 10, 0), 0);
    }
}
