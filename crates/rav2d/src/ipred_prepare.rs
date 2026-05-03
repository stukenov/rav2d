use crate::levels::*;

pub static MODE_CONV: [[[u8; 2]; 2]; 2] = [
    // DC_PRED
    [
        [DC_128_PRED, TOP_DC_PRED],
        [LEFT_DC_PRED, IntraPredMode::DcPred as u8],
    ],
    // PAETH_PRED
    [
        [DC_128_PRED, IntraPredMode::VertPred as u8],
        [IntraPredMode::HorPred as u8, IntraPredMode::PaethPred as u8],
    ],
];

#[derive(Clone, Copy, Default)]
pub struct EdgeMask {
    pub needs_left: bool,
    pub needs_top: bool,
    pub needs_topleft: bool,
    pub needs_topright: bool,
    pub needs_bottomleft: bool,
}

impl EdgeMask {
    const fn new(left: bool, top: bool, tl: bool, tr: bool, bl: bool) -> Self {
        Self {
            needs_left: left,
            needs_top: top,
            needs_topleft: tl,
            needs_topright: tr,
            needs_bottomleft: bl,
        }
    }
}

pub fn intra_prediction_edge(mode: u8) -> EdgeMask {
    match mode {
        0  /* DcPred */       => EdgeMask::new(true,  true,  false, false, false),
        1  /* VertPred */     => EdgeMask::new(false, true,  false, false, false),
        2  /* HorPred */      => EdgeMask::new(true,  false, false, false, false),
        _ if mode == LEFT_DC_PRED  => EdgeMask::new(true,  false, false, false, false),
        _ if mode == TOP_DC_PRED   => EdgeMask::new(false, true,  false, false, false),
        _ if mode == DC_128_PRED   => EdgeMask::new(false, false, false, false, false),
        _ if mode == Z1_PRED       => EdgeMask::new(false, true,  true,  true,  false),
        _ if mode == Z2_PRED       => EdgeMask::new(true,  true,  true,  false, false),
        _ if mode == Z3_PRED       => EdgeMask::new(true,  false, true,  false, true),
        9  /* SmoothPred */   => EdgeMask::new(true,  true,  false, true,  true),
        10 /* SmoothVPred */  => EdgeMask::new(false, true,  false, false, true),
        11 /* SmoothHPred */  => EdgeMask::new(true,  false, false, true,  false),
        12 /* PaethPred */    => EdgeMask::new(true,  true,  true,  false, false),
        _ if mode == DIP_PRED      => EdgeMask::new(true,  true,  true,  true,  true),
        _ => EdgeMask::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_conv_dc() {
        assert_eq!(MODE_CONV[0][0][0], DC_128_PRED);
        assert_eq!(MODE_CONV[0][0][1], TOP_DC_PRED);
        assert_eq!(MODE_CONV[0][1][0], LEFT_DC_PRED);
        assert_eq!(MODE_CONV[0][1][1], IntraPredMode::DcPred as u8);
    }

    #[test]
    fn test_mode_conv_paeth() {
        assert_eq!(MODE_CONV[1][0][0], DC_128_PRED);
        assert_eq!(MODE_CONV[1][0][1], IntraPredMode::VertPred as u8);
        assert_eq!(MODE_CONV[1][1][0], IntraPredMode::HorPred as u8);
        assert_eq!(MODE_CONV[1][1][1], IntraPredMode::PaethPred as u8);
    }

    #[test]
    fn test_edge_mask_dc_pred() {
        let e = intra_prediction_edge(IntraPredMode::DcPred as u8);
        assert!(e.needs_left && e.needs_top);
        assert!(!e.needs_topleft && !e.needs_topright && !e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_vert_pred() {
        let e = intra_prediction_edge(IntraPredMode::VertPred as u8);
        assert!(e.needs_top);
        assert!(!e.needs_left);
    }

    #[test]
    fn test_edge_mask_paeth_pred() {
        let e = intra_prediction_edge(IntraPredMode::PaethPred as u8);
        assert!(e.needs_left && e.needs_top && e.needs_topleft);
        assert!(!e.needs_topright && !e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_z1() {
        let e = intra_prediction_edge(Z1_PRED);
        assert!(e.needs_top && e.needs_topright && e.needs_topleft);
        assert!(!e.needs_left && !e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_dip() {
        let e = intra_prediction_edge(DIP_PRED);
        assert!(e.needs_left && e.needs_top && e.needs_topleft);
        assert!(e.needs_topright && e.needs_bottomleft);
    }

    #[test]
    fn test_edge_mask_dc128() {
        let e = intra_prediction_edge(DC_128_PRED);
        assert!(!e.needs_left && !e.needs_top && !e.needs_topleft);
    }

    #[test]
    fn test_edge_mask_unknown() {
        let e = intra_prediction_edge(255);
        assert!(!e.needs_left && !e.needs_top);
    }
}
