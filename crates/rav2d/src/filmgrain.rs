use crate::headers::FilmGrainData;
use crate::intops::iclip;
use crate::tables::GAUSSIAN_SEQUENCE;

pub const GRAIN_WIDTH: usize = 82;
pub const GRAIN_HEIGHT: usize = 73;
pub const SUB_GRAIN_WIDTH: usize = 44;
pub const SUB_GRAIN_HEIGHT: usize = 38;

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

pub fn generate_grain_y(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
) {
    let shift = 4 + data.grain_scale_shift;
    let grain_ctr = 128;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    for y in 0..GRAIN_HEIGHT {
        for x in 0..GRAIN_WIDTH {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = round2(GAUSSIAN_SEQUENCE[value] as i32, shift as u32) as i16;
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;

    for y in ar_pad..GRAIN_HEIGHT {
        for x in ar_pad..GRAIN_WIDTH - ar_pad {
            let coeff = &data.ar_coeffs[0];
            let mut sum = 0i32;
            let mut ci = 0usize;
            for dy in (y.wrapping_sub(ar_lag))..=y {
                let dx_start = if dy == y { x.wrapping_sub(ar_lag) } else { x.wrapping_sub(ar_lag) };
                let dx_end = if dy == y { x } else { x + ar_lag + 1 };
                for dx in dx_start..dx_end {
                    if dy == y && dx == x {
                        break;
                    }
                    sum += coeff[ci] as i32 * buf[dy][dx] as i32;
                    ci += 1;
                }
            }

            let grain = buf[y][x] as i32 + round2(sum, data.ar_coeff_shift as u32);
            buf[y][x] = iclip(grain, grain_min, grain_max) as i16;
        }
    }
}

pub fn generate_grain_uv(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    buf_y: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
    uv: usize,
    subx: bool,
    suby: bool,
) {
    seed ^= if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = 4 + data.grain_scale_shift;
    let grain_ctr = 128;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let chroma_w = if subx { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH };
    let chroma_h = if suby { SUB_GRAIN_HEIGHT } else { GRAIN_HEIGHT };

    for y in 0..chroma_h {
        for x in 0..chroma_w {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = round2(GAUSSIAN_SEQUENCE[value] as i32, shift as u32) as i16;
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    let subx_i = subx as usize;
    let suby_i = suby as usize;

    for y in ar_pad..chroma_h {
        for x in ar_pad..chroma_w - ar_pad {
            let coeff = &data.ar_coeffs[1 + uv];
            let mut sum = 0i32;
            let mut ci = 0usize;
            'outer: for dy in (y.wrapping_sub(ar_lag))..=y {
                let dx_start = x.wrapping_sub(ar_lag);
                let dx_end = if dy == y { x } else { x + ar_lag + 1 };
                for dx in dx_start..dx_end {
                    if dy == y && dx == x {
                        if data.num_points[0] > 0 {
                            let luma_x = ((x - ar_pad) << subx_i) + ar_pad;
                            let luma_y = ((y - ar_pad) << suby_i) + ar_pad;
                            let mut luma = 0i32;
                            for i in 0..=suby_i {
                                for j in 0..=subx_i {
                                    luma += buf_y[luma_y + i][luma_x + j] as i32;
                                }
                            }
                            luma = round2(luma, (subx_i + suby_i) as u32);
                            sum += luma * coeff[ci] as i32;
                        }
                        break 'outer;
                    }
                    sum += coeff[ci] as i32 * buf[dy][dx] as i32;
                    ci += 1;
                }
            }

            let grain = buf[y][x] as i32 + round2(sum, data.ar_coeff_shift as u32);
            buf[y][x] = iclip(grain, grain_min, grain_max) as i16;
        }
    }
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

    fn make_grain_data(ar_lag: i32, grain_scale_shift: i32) -> FilmGrainData {
        let mut d = FilmGrainData::default();
        d.ar_coeff_lag = ar_lag;
        d.grain_scale_shift = grain_scale_shift;
        d.ar_coeff_shift = 6;
        d.num_points = [0, 0, 0];
        d
    }

    #[test]
    fn test_generate_grain_y_no_ar() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_y(&mut buf, &data, 1234);
        let nonzero = buf.iter().flat_map(|r| r.iter()).any(|&v| v != 0);
        assert!(nonzero);
    }

    #[test]
    fn test_generate_grain_y_bounded() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_y(&mut buf, &data, 5678);
        for row in &buf {
            for &v in row.iter() {
                assert!(v >= -128 && v <= 127, "grain value {} out of 8bpc range", v);
            }
        }
    }

    #[test]
    fn test_generate_grain_y_deterministic() {
        let mut buf1 = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let mut buf2 = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(1, 0);
        generate_grain_y(&mut buf1, &data, 42);
        generate_grain_y(&mut buf2, &data, 42);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_generate_grain_y_ar_lag1() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let mut data = make_grain_data(1, 0);
        data.ar_coeffs[0][0] = 10;
        data.ar_coeffs[0][1] = -5;
        data.ar_coeffs[0][2] = 3;
        generate_grain_y(&mut buf, &data, 999);
        for row in &buf {
            for &v in row.iter() {
                assert!(v >= -128 && v <= 127);
            }
        }
    }

    #[test]
    fn test_generate_grain_uv_no_sub() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let buf_y = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_uv(&mut buf, &buf_y, &data, 100, 0, false, false);
        let nonzero = buf.iter().flat_map(|r| r.iter()).any(|&v| v != 0);
        assert!(nonzero);
    }

    #[test]
    fn test_generate_grain_uv_with_sub() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let buf_y = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_uv(&mut buf, &buf_y, &data, 100, 0, true, true);
        for y in 0..SUB_GRAIN_HEIGHT {
            for x in 0..SUB_GRAIN_WIDTH {
                assert!(buf[y][x] >= -128 && buf[y][x] <= 127);
            }
        }
    }

    #[test]
    fn test_generate_grain_uv_seed_xor() {
        let mut buf_u = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let mut buf_v = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let buf_y = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_uv(&mut buf_u, &buf_y, &data, 100, 0, false, false);
        generate_grain_uv(&mut buf_v, &buf_y, &data, 100, 1, false, false);
        assert_ne!(buf_u[0][0], buf_v[0][0]);
    }

    #[test]
    fn test_generate_grain_uv_with_luma_correlation() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let buf_y = [[64i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let mut data = make_grain_data(1, 0);
        data.num_points[0] = 1;
        data.ar_coeffs[1][0] = 5;
        data.ar_coeffs[1][1] = -3;
        data.ar_coeffs[1][2] = 2;
        generate_grain_uv(&mut buf, &buf_y, &data, 42, 0, false, false);
        for row in &buf[..GRAIN_HEIGHT] {
            for &v in row[..GRAIN_WIDTH].iter() {
                assert!(v >= -128 && v <= 127);
            }
        }
    }
}
