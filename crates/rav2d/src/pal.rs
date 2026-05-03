pub fn pal_idx_finish(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    debug_assert!(bw >= 4 && bw <= 64 && (bw & (bw - 1)) == 0);
    debug_assert!(bh >= 4 && bh <= 64 && (bh & (bh - 1)) == 0);
    debug_assert!(w >= 4 && w <= bw && (w & 3) == 0);
    debug_assert!(h >= 4 && h <= bh && (h & 3) == 0);

    let dst_w = w / 2;
    let dst_bw = bw / 2;

    for y in 0..h {
        let src_row = &src[y * bw..];
        let dst_row = &mut dst[y * dst_bw..];
        for x in 0..dst_w {
            dst_row[x] = src_row[x * 2] | (src_row[x * 2 + 1] << 4);
        }
        if dst_w < dst_bw {
            let fill = src_row[w - 1] * 0x11;
            dst_row[dst_w..dst_bw].fill(fill);
        }
    }

    if h < bh {
        let last_row_start = (h - 1) * dst_bw;
        for y in h..bh {
            let row_start = y * dst_bw;
            for x in 0..dst_bw {
                dst[row_start + x] = dst[last_row_start + x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pal_idx_finish_basic() {
        let src: Vec<u8> = (0..16).collect();
        let mut dst = vec![0u8; 8];
        pal_idx_finish(&mut dst, &src, 4, 4, 4, 4);
        assert_eq!(dst[0], 0 | (1 << 4));
        assert_eq!(dst[1], 2 | (3 << 4));
    }

    #[test]
    fn test_pal_idx_finish_edge_fill() {
        let src = vec![1u8; 64];
        let mut dst = vec![0u8; 32];
        pal_idx_finish(&mut dst, &src, 8, 8, 4, 4);
        // visible area packed
        assert_eq!(dst[0], 1 | (1 << 4));
        // right edge fill: 1*0x11 = 0x11
        assert_eq!(dst[2], 0x11);
        // bottom edge fill: copy of last visible row
        assert_eq!(dst[4 * 4], dst[3 * 4]);
    }

    #[test]
    fn test_pal_idx_finish_full_block() {
        let src = vec![3u8; 16];
        let mut dst = vec![0u8; 8];
        pal_idx_finish(&mut dst, &src, 4, 4, 4, 4);
        for b in &dst {
            assert_eq!(*b, 3 | (3 << 4));
        }
    }
}
