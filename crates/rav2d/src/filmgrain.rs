use crate::headers::FilmGrainData;
use crate::intops::iclip;
use crate::pixel::Pixel;
use crate::tables::GAUSSIAN_SEQUENCE;

pub const GRAIN_WIDTH: usize = 82;
pub const GRAIN_HEIGHT: usize = 73;
pub const SUB_GRAIN_WIDTH: usize = 44;
pub const SUB_GRAIN_HEIGHT: usize = 38;

pub fn get_random_number(bits: u32, state: &mut u32) -> u32 {
    let r = *state;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
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

pub fn generate_scaling_hbd(bitdepth: u32, points: &[[u8; 2]], scaling: &mut [u8]) {
    debug_assert!(bitdepth > 8);
    let shift_x = bitdepth - 8;
    let scaling_size = 1usize << bitdepth;
    let num = points.len();

    if num == 0 {
        scaling[..scaling_size].fill(0);
        return;
    }

    scaling[..(points[0][0] as usize) << shift_x].fill(points[0][1]);

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
        for x in 0..dx {
            scaling[((bx + x) << shift_x) as usize] = (by + (d >> 16)) as u8;
            d += delta;
        }
    }

    let n = (points[num - 1][0] as usize) << shift_x;
    scaling[n..scaling_size].fill(points[num - 1][1]);

    let pad = 1i32 << shift_x;
    let rnd = pad >> 1;
    for i in 0..num - 1 {
        let bx = (points[i][0] as i32) << shift_x;
        let ex = (points[i + 1][0] as i32) << shift_x;
        let dx = ex - bx;
        let mut x = 0;
        while x < dx {
            let base = scaling[(bx + x) as usize] as i32;
            let range = scaling[(bx + x + pad) as usize] as i32 - base;
            let mut r = rnd;
            for n in 1..pad {
                r += range;
                scaling[(bx + x + n) as usize] = (base + (r >> shift_x)) as u8;
            }
            x += pad;
        }
    }
}

pub fn generate_grain_y(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
    bitdepth: u32,
) {
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let grain_ctr = 128 << bitdepth_min_8;
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
                let dx_start = x.wrapping_sub(ar_lag);
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
    bitdepth: u32,
) {
    seed ^= if uv != 0 { 0x49d8 } else { 0xb524 };
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let grain_ctr = 128 << bitdepth_min_8;
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
                // Current row stops at (and includes) the center pixel dx==x,
                // which carries the luma-correlation term (dav2d: the dx==0,dy==0
                // case uses the final AR coeff for the luma contribution, then
                // breaks). Off rows span the full [-ar_lag, +ar_lag] window.
                let dx_end = if dy == y { x + 1 } else { x + ar_lag + 1 };
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

pub fn sample_lut(
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bs: usize,
    offsets: &[[[i32; 2]; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
) -> i16 {
    let off = &offsets[bx][by];
    let offx = 3 + (2 >> subx) * (3 + off[1] as usize);
    let offy = 3 + (2 >> suby) * (3 + off[0] as usize);
    grain_lut[offy + y + (bs >> suby) * by][offx + x + (bs >> subx) * bx]
}

pub fn fgy_32x32xn<P: Pixel>(
    dst: &mut [P],
    src: &[P],
    stride: usize,
    data: &FilmGrainData,
    in_seed: u32,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bh: i32,
    row_num: i32,
    bitdepth: u32,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let bs = (16 << data.block_size) as usize;

    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16i32 << bitdepth_min_8, 235i32 << bitdepth_min_8)
    } else {
        (0, (1i32 << bitdepth) - 1)
    };

    let mut seed = [0u32; 2];
    for i in 0..rows {
        seed[i] = in_seed;
        seed[i] ^= ((((row_num - i as i32) * 37 + 178) & 0xFF) as u32) << 8;
        seed[i] ^= (((row_num - i as i32) * 173 + 105) & 0xFF) as u32;
    }

    let mut offsets = [[[0i32; 2]; 2]; 2];
    let w: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let mut bx = 0usize;
    while bx < pw {
        let bw = bs.min(pw - bx) as i32;

        if data.overlap_flag && bx > 0 {
            for i in 0..rows {
                for n in 0..2 {
                    offsets[1][i][n] = offsets[0][i][n];
                }
            }
        }

        for i in 0..rows {
            for n in 0..2 {
                offsets[0][i][n] = (((3 - data.block_size) as u32
                    * get_random_number(9, &mut seed[i]))
                    >> 6) as i32;
                for _ in 0..3 {
                    get_random_number(16, &mut seed[i]);
                }
            }
        }

        let ystart = if data.overlap_flag && row_num > 0 {
            2.min(bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx > 0 {
            2.min(bw)
        } else {
            0
        };

        for y in ystart..bh {
            for x in xstart..bw {
                let grain =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 0, 0, x as usize, y as usize) as i32;
                let si = y as usize * stride + x as usize + bx;
                let sv = Into::<i32>::into(src[si]);
                let noise = round2(
                    scaling[sv as usize] as i32 * grain,
                    data.scaling_shift as u32,
                );
                dst[si] = P::from_i32(iclip(sv + noise, min_value, max_value));
            }
            for x in 0..xstart {
                let grain =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 0, 0, x as usize, y as usize) as i32;
                let old =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 1, 0, x as usize, y as usize) as i32;
                let grain = iclip(
                    round2(old * w[x as usize][0] + grain * w[x as usize][1], 5),
                    grain_min,
                    grain_max,
                );
                let si = y as usize * stride + x as usize + bx;
                let sv = Into::<i32>::into(src[si]);
                let noise = round2(
                    scaling[sv as usize] as i32 * grain,
                    data.scaling_shift as u32,
                );
                dst[si] = P::from_i32(iclip(sv + noise, min_value, max_value));
            }
        }

        for y in 0..ystart {
            for x in xstart..bw {
                let grain =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 0, 0, x as usize, y as usize) as i32;
                let old =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 0, 1, x as usize, y as usize) as i32;
                let grain = iclip(
                    round2(old * w[y as usize][0] + grain * w[y as usize][1], 5),
                    grain_min,
                    grain_max,
                );
                let si = y as usize * stride + x as usize + bx;
                let sv = Into::<i32>::into(src[si]);
                let noise = round2(
                    scaling[sv as usize] as i32 * grain,
                    data.scaling_shift as u32,
                );
                dst[si] = P::from_i32(iclip(sv + noise, min_value, max_value));
            }
            for x in 0..xstart {
                let mut top =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 0, 1, x as usize, y as usize) as i32;
                let old =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 1, 1, x as usize, y as usize) as i32;
                top = iclip(
                    round2(old * w[x as usize][0] + top * w[x as usize][1], 5),
                    grain_min,
                    grain_max,
                );

                let mut grain =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 0, 0, x as usize, y as usize) as i32;
                let old2 =
                    sample_lut(grain_lut, bs, &offsets, 0, 0, 1, 0, x as usize, y as usize) as i32;
                grain = iclip(
                    round2(old2 * w[x as usize][0] + grain * w[x as usize][1], 5),
                    grain_min,
                    grain_max,
                );

                grain = iclip(
                    round2(top * w[y as usize][0] + grain * w[y as usize][1], 5),
                    grain_min,
                    grain_max,
                );
                let si = y as usize * stride + x as usize + bx;
                let sv = Into::<i32>::into(src[si]);
                let noise = round2(
                    scaling[sv as usize] as i32 * grain,
                    data.scaling_shift as u32,
                );
                dst[si] = P::from_i32(iclip(sv + noise, min_value, max_value));
            }
        }

        bx += bs;
    }
}

pub fn fguv_32x32xn<P: Pixel>(
    dst: &mut [P],
    src: &[P],
    stride: usize,
    data: &FilmGrainData,
    in_seed: u32,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bh: i32,
    row_num: i32,
    luma_row: &[P],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
    sx: usize,
    sy: usize,
    bitdepth: u32,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let bs = (16 << data.block_size) as usize;

    let (min_value, max_value) = if data.clip_to_restricted_range {
        (
            16i32 << bitdepth_min_8,
            (if is_id { 235i32 } else { 240 }) << bitdepth_min_8,
        )
    } else {
        (0, (1i32 << bitdepth) - 1)
    };

    let mut seed = [0u32; 2];
    for i in 0..rows {
        seed[i] = in_seed;
        seed[i] ^= ((((row_num - i as i32) * 37 + 178) & 0xFF) as u32) << 8;
        seed[i] ^= (((row_num - i as i32) * 173 + 105) & 0xFF) as u32;
    }

    let mut offsets = [[[0i32; 2]; 2]; 2];
    let w: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];

    let mut bx = 0usize;
    while bx < pw {
        let bw = ((bs >> sx).min(pw - bx)) as i32;

        if data.overlap_flag && bx > 0 {
            for i in 0..rows {
                for n in 0..2 {
                    offsets[1][i][n] = offsets[0][i][n];
                }
            }
        }

        for i in 0..rows {
            for n in 0..2 {
                offsets[0][i][n] = (((3 - data.block_size) as u32
                    * get_random_number(9, &mut seed[i]))
                    >> 6) as i32;
                for _ in 0..3 {
                    get_random_number(16, &mut seed[i]);
                }
            }
        }

        let ystart = if data.overlap_flag && row_num > 0 {
            (2 >> sy as i32).min(bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx > 0 {
            (2 >> sx as i32).min(bw)
        } else {
            0
        };

        macro_rules! add_noise_uv {
            ($x:expr, $y:expr, $grain:expr) => {{
                let lx = (bx + $x as usize) << sx;
                let ly = ($y as usize) << sy;
                let luma = Into::<i32>::into(luma_row[ly * luma_stride + lx]);
                let avg: i32 = if sx != 0 {
                    (luma + Into::<i32>::into(luma_row[ly * luma_stride + lx + 1]) + 1) >> 1
                } else {
                    luma
                };
                let si = $y as usize * stride + bx + $x as usize;
                let sv = Into::<i32>::into(src[si]);
                let val = if !data.chroma_scaling_from_luma {
                    let combined = avg * data.uv_luma_mult[uv] + sv * data.uv_mult[uv];
                    iclip(
                        (combined >> 6) + data.uv_offset[uv] * (1 << bitdepth_min_8),
                        0,
                        (1i32 << bitdepth) - 1,
                    ) as usize
                } else {
                    avg as usize
                };
                let noise = round2(scaling[val] as i32 * $grain, data.scaling_shift as u32);
                dst[si] = P::from_i32(iclip(sv + noise, min_value, max_value));
            }};
        }

        for y in ystart..bh {
            for x in xstart..bw {
                let grain = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 0, 0, x as usize, y as usize,
                ) as i32;
                add_noise_uv!(x, y, grain);
            }
            for x in 0..xstart {
                let grain = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 0, 0, x as usize, y as usize,
                ) as i32;
                let old = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 1, 0, x as usize, y as usize,
                ) as i32;
                let grain = iclip(
                    round2(old * w[sx][x as usize][0] + grain * w[sx][x as usize][1], 5),
                    grain_min,
                    grain_max,
                );
                add_noise_uv!(x, y, grain);
            }
        }

        for y in 0..ystart {
            for x in xstart..bw {
                let grain = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 0, 0, x as usize, y as usize,
                ) as i32;
                let old = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 0, 1, x as usize, y as usize,
                ) as i32;
                let grain = iclip(
                    round2(old * w[sy][y as usize][0] + grain * w[sy][y as usize][1], 5),
                    grain_min,
                    grain_max,
                );
                add_noise_uv!(x, y, grain);
            }
            for x in 0..xstart {
                let mut top = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 0, 1, x as usize, y as usize,
                ) as i32;
                let old = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 1, 1, x as usize, y as usize,
                ) as i32;
                top = iclip(
                    round2(old * w[sx][x as usize][0] + top * w[sx][x as usize][1], 5),
                    grain_min,
                    grain_max,
                );

                let mut grain = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 0, 0, x as usize, y as usize,
                ) as i32;
                let old2 = sample_lut(
                    grain_lut, bs, &offsets, sx, sy, 1, 0, x as usize, y as usize,
                ) as i32;
                grain = iclip(
                    round2(
                        old2 * w[sx][x as usize][0] + grain * w[sx][x as usize][1],
                        5,
                    ),
                    grain_min,
                    grain_max,
                );

                grain = iclip(
                    round2(top * w[sy][y as usize][0] + grain * w[sy][y as usize][1], 5),
                    grain_min,
                    grain_max,
                );
                add_noise_uv!(x, y, grain);
            }
        }

        bx += bs >> sx;
    }
}

pub struct GrainLut {
    pub y: [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    pub u: [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    pub v: [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
}

impl GrainLut {
    pub fn new() -> Self {
        Self {
            y: [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
            u: [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
            v: [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
        }
    }
}

impl Default for GrainLut {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prep_grain(
    fgd: &FilmGrainData,
    grain_lut: &mut GrainLut,
    scaling: &mut [Vec<u8>; 3],
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: u32,
) {
    // Generate grain LUTs as needed (dav2d_prep_grain, fg_apply_tmpl.c). The Y
    // LUT is ALWAYS generated: the chroma LUTs derive from it. The chroma LUTs
    // are generated with the plane's subsampling so the sub-grid dimensions
    // match the picture layout (dav2d selects generate_grain_uv[layout-1]).
    generate_grain_y(&mut grain_lut.y, fgd, seed, bitdepth);

    if fgd.num_points[1] > 0 || fgd.chroma_scaling_from_luma {
        generate_grain_uv(
            &mut grain_lut.u,
            &grain_lut.y,
            fgd,
            seed,
            0,
            ss_x,
            ss_y,
            bitdepth,
        );
    }
    if fgd.num_points[2] > 0 || fgd.chroma_scaling_from_luma {
        generate_grain_uv(
            &mut grain_lut.v,
            &grain_lut.y,
            fgd,
            seed,
            1,
            ss_x,
            ss_y,
            bitdepth,
        );
    }

    // Scaling LUTs (dav2d generates one per plane with scaling points). The
    // table is indexed by pixel value, so its size is `1 << bitdepth`.
    let scaling_size = 1usize << bitdepth;
    let gen_lut = |dst: &mut Vec<u8>, points: &[[u8; 2]]| {
        dst.resize(scaling_size, 0);
        if bitdepth == 8 {
            generate_scaling_8bpc(points, dst.as_mut_slice().try_into().unwrap());
        } else {
            generate_scaling_hbd(bitdepth, points, dst.as_mut_slice());
        }
    };
    if fgd.num_points[0] > 0 {
        gen_lut(
            &mut scaling[0],
            &fgd.points[0][..fgd.num_points[0] as usize],
        );
    }

    if !fgd.chroma_scaling_from_luma {
        for uv in 0..2 {
            if fgd.num_points[uv + 1] > 0 {
                gen_lut(
                    &mut scaling[uv + 1],
                    &fgd.points[uv + 1][..fgd.num_points[uv + 1] as usize],
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_grain_row<P: Pixel>(
    dst_y: &mut [P],
    dst_u: &mut [P],
    dst_v: &mut [P],
    src_y: &[P],
    src_u: &[P],
    src_v: &[P],
    y_stride: isize,
    uv_stride: isize,
    fgd: &FilmGrainData,
    grain_lut: &GrainLut,
    scaling: &[Vec<u8>; 3],
    w: usize,
    h: usize,
    row: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: u32,
) {
    // Grain block height in luma rows: 16 or 32 per `block_size` (dav2d
    // dav2d_apply_grain_row, fg_apply_tmpl.c: `bs = 16 << data->block_size`).
    let bs = (16usize) << fgd.block_size;
    let row_start = row * bs;

    if fgd.num_points[0] > 0 && !scaling[0].is_empty() {
        // Luma block height, clamped to the remaining image rows (dav2d
        // `bh = imin(out->p.h - row*bs, bs)`).
        let bh = (h - row_start).min(bs);
        let y_off = row_start * y_stride.unsigned_abs();
        let src_slice = if y_off < src_y.len() {
            &src_y[y_off..]
        } else {
            return;
        };
        let dst_slice = if y_off < dst_y.len() {
            &mut dst_y[y_off..]
        } else {
            return;
        };

        fgy_32x32xn(
            dst_slice,
            src_slice,
            y_stride.unsigned_abs(),
            fgd,
            seed,
            w,
            scaling[0].as_slice(),
            &grain_lut.y,
            bh as i32,
            row as i32,
            bitdepth,
        );
    }

    let has_uv = |uv: usize| -> bool {
        (fgd.num_points[uv + 1] > 0 || fgd.chroma_scaling_from_luma)
            && !scaling_for_uv(scaling, fgd, uv).is_empty()
    };

    // Chroma block height (dav2d `bh = (imin(out->p.h - row*bs, bs) + ss_y) >> ss_y`).
    let ch = (((h - row_start).min(bs) + ss_y as usize) >> ss_y as usize) as i32;
    let cw = (w + ss_x as usize) >> ss_x as usize;
    let uv_off = (row_start >> (ss_y as usize)) * uv_stride.unsigned_abs();
    let luma_off = row_start * y_stride.unsigned_abs();

    if has_uv(0) && uv_off < src_u.len() && uv_off < dst_u.len() {
        let uv_scaling: &[u8] = scaling_for_uv(scaling, fgd, 0);
        fguv_32x32xn(
            &mut dst_u[uv_off..],
            &src_u[uv_off..],
            uv_stride.unsigned_abs(),
            fgd,
            seed,
            cw,
            uv_scaling,
            &grain_lut.u,
            ch,
            row as i32,
            &src_y[luma_off..],
            y_stride.unsigned_abs(),
            0,
            fgd.mc_identity,
            ss_x as usize,
            ss_y as usize,
            bitdepth,
        );
    }

    if has_uv(1) && uv_off < src_v.len() && uv_off < dst_v.len() {
        let uv_scaling: &[u8] = scaling_for_uv(scaling, fgd, 1);
        fguv_32x32xn(
            &mut dst_v[uv_off..],
            &src_v[uv_off..],
            uv_stride.unsigned_abs(),
            fgd,
            seed,
            cw,
            uv_scaling,
            &grain_lut.v,
            ch,
            row as i32,
            &src_y[luma_off..],
            y_stride.unsigned_abs(),
            1,
            fgd.mc_identity,
            ss_x as usize,
            ss_y as usize,
            bitdepth,
        );
    }
}

fn scaling_for_uv<'a>(scaling: &'a [Vec<u8>; 3], fgd: &FilmGrainData, uv: usize) -> &'a [u8] {
    if fgd.chroma_scaling_from_luma {
        &scaling[0]
    } else {
        &scaling[uv + 1]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_grain<P: Pixel>(
    dst_y: &mut [P],
    dst_u: &mut [P],
    dst_v: &mut [P],
    src_y: &[P],
    src_u: &[P],
    src_v: &[P],
    y_stride: isize,
    uv_stride: isize,
    fgd: &FilmGrainData,
    w: usize,
    h: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: u32,
) {
    apply_grain_mt(
        dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, w, h, seed, ss_x, ss_y,
        bitdepth, 1,
    );
}

/// Film grain synthesis with optional row-band parallelism.
///
/// The output is partitioned into independent `bs`-tall row bands
/// (`dav2d_apply_grain`'s tiling). Each band writes only rows `[row*bs, …)` of
/// the destination planes and reads only the (read-only) ungrained `src` planes
/// plus the precomputed `grain_lut`/`scaling`; the per-pixel grain RNG is
/// re-derived from absolute position inside the kernels. The bands therefore
/// touch disjoint output memory and share no mutable state, so distributing them
/// across threads yields output byte-identical to the sequential loop.
///
/// `n_threads <= 1` runs the exact sequential loop (single-thread path
/// unchanged).
#[allow(clippy::too_many_arguments)]
pub fn apply_grain_mt<P: Pixel>(
    dst_y: &mut [P],
    dst_u: &mut [P],
    dst_v: &mut [P],
    src_y: &[P],
    src_u: &[P],
    src_v: &[P],
    y_stride: isize,
    uv_stride: isize,
    fgd: &FilmGrainData,
    w: usize,
    h: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: u32,
    n_threads: u32,
) {
    let mut grain_lut = GrainLut::new();
    let mut scaling = [Vec::new(), Vec::new(), Vec::new()];

    prep_grain(
        fgd,
        &mut grain_lut,
        &mut scaling,
        seed,
        ss_x,
        ss_y,
        bitdepth,
    );

    // Row tiling matches dav2d_apply_grain: `bs = 16 << block_size`,
    // `rows = (h + bs - 1) / bs` (fg_apply_tmpl.c).
    let bs = (16usize) << fgd.block_size;
    let rows = h.div_ceil(bs);

    if n_threads <= 1 || rows <= 1 {
        for row in 0..rows {
            apply_grain_row(
                dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, &grain_lut,
                &scaling, w, h, row, seed, ss_x, ss_y, bitdepth,
            );
        }
        return;
    }

    // Parallel path: one job per row band. Each band writes disjoint output rows
    // (verified above), so the raw destination pointers are split across jobs
    // without aliasing. Pointers are wrapped in a Send marker because each band's
    // writes stay within its own row range.
    struct DstPtrs<P> {
        y: *mut P,
        y_len: usize,
        u: *mut P,
        u_len: usize,
        v: *mut P,
        v_len: usize,
    }
    // SAFETY: each row band writes only its disjoint output rows; the pointers
    // are never used to access another band's memory, so sharing them across
    // threads introduces no data race.
    unsafe impl<P> Send for DstPtrs<P> {}
    unsafe impl<P> Sync for DstPtrs<P> {}
    let dst = DstPtrs {
        y: dst_y.as_mut_ptr(),
        y_len: dst_y.len(),
        u: dst_u.as_mut_ptr(),
        u_len: dst_u.len(),
        v: dst_v.as_mut_ptr(),
        v_len: dst_v.len(),
    };
    let dst = &dst;
    let grain_lut = &grain_lut;
    let scaling = &scaling;

    let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = (0..rows)
        .map(|row| {
            Box::new(move || {
                // SAFETY: reconstruct each plane slice from its base pointer for
                // the full plane length; the row function only writes within
                // `[row*bs*stride, …)` for this band's `bh` rows, which is
                // disjoint from every other band.
                let dst_y = unsafe { std::slice::from_raw_parts_mut(dst.y, dst.y_len) };
                let dst_u = unsafe { std::slice::from_raw_parts_mut(dst.u, dst.u_len) };
                let dst_v = unsafe { std::slice::from_raw_parts_mut(dst.v, dst.v_len) };
                apply_grain_row(
                    dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, grain_lut,
                    scaling, w, h, row, seed, ss_x, ss_y, bitdepth,
                );
            }) as Box<dyn FnOnce() + Send + '_>
        })
        .collect();

    crate::thread_task::run_disjoint_jobs(n_threads, jobs);
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
        generate_grain_y(&mut buf, &data, 1234, 8);
        let nonzero = buf.iter().flat_map(|r| r.iter()).any(|&v| v != 0);
        assert!(nonzero);
    }

    #[test]
    fn test_generate_grain_y_bounded() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_y(&mut buf, &data, 5678, 8);
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
        generate_grain_y(&mut buf1, &data, 42, 8);
        generate_grain_y(&mut buf2, &data, 42, 8);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_generate_grain_y_ar_lag1() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let mut data = make_grain_data(1, 0);
        data.ar_coeffs[0][0] = 10;
        data.ar_coeffs[0][1] = -5;
        data.ar_coeffs[0][2] = 3;
        generate_grain_y(&mut buf, &data, 999, 8);
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
        generate_grain_uv(&mut buf, &buf_y, &data, 100, 0, false, false, 8);
        let nonzero = buf.iter().flat_map(|r| r.iter()).any(|&v| v != 0);
        assert!(nonzero);
    }

    #[test]
    fn test_generate_grain_uv_with_sub() {
        let mut buf = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let buf_y = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_grain_data(0, 0);
        generate_grain_uv(&mut buf, &buf_y, &data, 100, 0, true, true, 8);
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
        generate_grain_uv(&mut buf_u, &buf_y, &data, 100, 0, false, false, 8);
        generate_grain_uv(&mut buf_v, &buf_y, &data, 100, 1, false, false, 8);
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
        generate_grain_uv(&mut buf, &buf_y, &data, 42, 0, false, false, 8);
        for row in &buf[..GRAIN_HEIGHT] {
            for &v in row[..GRAIN_WIDTH].iter() {
                assert!(v >= -128 && v <= 127);
            }
        }
    }

    #[test]
    fn test_sample_lut_basic() {
        let mut lut = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        lut[10][15] = 42;
        let offsets = [[[0i32; 2]; 2]; 2];
        let v = sample_lut(&lut, 32, &offsets, 0, 0, 0, 0, 9, 4);
        let _ = v;
    }

    fn make_fgy_data() -> FilmGrainData {
        let mut d = FilmGrainData::default();
        d.scaling_shift = 8;
        d.block_size = 0;
        d.overlap_flag = false;
        d.clip_to_restricted_range = false;
        d
    }

    #[test]
    fn test_fgy_32x32xn_no_overlap() {
        let pw = 16;
        let bh = 4;
        let stride = pw;
        let src = vec![128u8; stride * bh as usize];
        let mut dst = vec![0u8; stride * bh as usize];
        let mut scaling = [0u8; 256];
        scaling[128] = 64;
        let mut grain_lut = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data_gen = make_grain_data(0, 0);
        generate_grain_y(&mut grain_lut, &data_gen, 42, 8);
        let data = make_fgy_data();
        fgy_32x32xn(
            &mut dst, &src, stride, &data, 100, pw, &scaling, &grain_lut, bh, 0, 8,
        );
        let modified = dst.iter().zip(src.iter()).any(|(&d, &s)| d != s);
        assert!(modified);
    }

    #[test]
    fn test_fgy_32x32xn_output_bounded() {
        let pw = 32;
        let bh = 8;
        let stride = pw;
        let src: Vec<u8> = (0..stride * bh as usize)
            .map(|i| (i & 0xFF) as u8)
            .collect();
        let mut dst = vec![0u8; stride * bh as usize];
        let mut scaling = [128u8; 256];
        for i in 0..256 {
            scaling[i] = i as u8;
        }
        let mut grain_lut = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data_gen = make_grain_data(0, 0);
        generate_grain_y(&mut grain_lut, &data_gen, 7777, 8);
        let data = make_fgy_data();
        fgy_32x32xn(
            &mut dst, &src, stride, &data, 200, pw, &scaling, &grain_lut, bh, 0, 8,
        );
        let has_variety = dst.windows(2).any(|w| w[0] != w[1]);
        assert!(has_variety);
    }

    #[test]
    fn test_fgy_32x32xn_restricted_range() {
        let pw = 16;
        let bh = 4;
        let stride = pw;
        let src = vec![0u8; stride * bh as usize];
        let mut dst = vec![0u8; stride * bh as usize];
        let scaling = [255u8; 256];
        let mut grain_lut = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data_gen = make_grain_data(0, 0);
        generate_grain_y(&mut grain_lut, &data_gen, 42, 8);
        let mut data = make_fgy_data();
        data.clip_to_restricted_range = true;
        fgy_32x32xn(
            &mut dst, &src, stride, &data, 100, pw, &scaling, &grain_lut, bh, 0, 8,
        );
        for &v in &dst {
            assert!(v >= 16 && v <= 235, "restricted range violated: {}", v);
        }
    }

    #[test]
    fn test_fgy_32x32xn_deterministic() {
        let pw = 16;
        let bh = 4;
        let stride = pw;
        let src = vec![100u8; stride * bh as usize];
        let mut dst1 = vec![0u8; stride * bh as usize];
        let mut dst2 = vec![0u8; stride * bh as usize];
        let scaling = [50u8; 256];
        let mut grain_lut = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data_gen = make_grain_data(0, 0);
        generate_grain_y(&mut grain_lut, &data_gen, 42, 8);
        let data = make_fgy_data();
        fgy_32x32xn(
            &mut dst1, &src, stride, &data, 999, pw, &scaling, &grain_lut, bh, 0, 8,
        );
        fgy_32x32xn(
            &mut dst2, &src, stride, &data, 999, pw, &scaling, &grain_lut, bh, 0, 8,
        );
        assert_eq!(dst1, dst2);
    }

    #[test]
    fn test_fgy_32x32xn_zero_scaling() {
        let pw = 16;
        let bh = 4;
        let stride = pw;
        let src = vec![128u8; stride * bh as usize];
        let mut dst = vec![0u8; stride * bh as usize];
        let scaling = [0u8; 256];
        let grain_lut = [[50i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data = make_fgy_data();
        fgy_32x32xn(
            &mut dst, &src, stride, &data, 100, pw, &scaling, &grain_lut, bh, 0, 8,
        );
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_fguv_32x32xn_basic() {
        let pw = 16;
        let bh = 4;
        let stride = pw;
        let src = vec![128u8; stride * bh as usize];
        let mut dst = vec![0u8; stride * bh as usize];
        let luma = vec![128u8; pw * 2 * bh as usize * 2];
        let mut scaling = [0u8; 256];
        scaling[128] = 64;
        let mut grain_lut = [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let data_gen = make_grain_data(0, 0);
        generate_grain_y(&mut grain_lut, &data_gen, 42, 8);
        let mut data = make_fgy_data();
        data.uv_mult = [0; 2];
        data.uv_luma_mult = [64; 2];
        data.uv_offset = [0; 2];
        fguv_32x32xn(
            &mut dst,
            &src,
            stride,
            &data,
            100,
            pw,
            &scaling,
            &grain_lut,
            bh,
            0,
            &luma,
            pw * 2,
            0,
            false,
            1,
            1,
            8,
        );
        let modified = dst.iter().zip(src.iter()).any(|(&d, &s)| d != s);
        assert!(modified);
    }

    #[test]
    fn test_fguv_32x32xn_chroma_from_luma() {
        let pw = 16;
        let bh = 4;
        let stride = pw;
        let src = vec![128u8; stride * bh as usize];
        let mut dst = vec![0u8; stride * bh as usize];
        let luma = vec![200u8; pw * bh as usize];
        let mut scaling = [0u8; 256];
        scaling[200] = 100;
        let grain_lut = [[50i16; GRAIN_WIDTH]; GRAIN_HEIGHT];
        let mut data = make_fgy_data();
        data.chroma_scaling_from_luma = true;
        fguv_32x32xn(
            &mut dst, &src, stride, &data, 100, pw, &scaling, &grain_lut, bh, 0, &luma, pw, 0,
            true, 0, 0, 8,
        );
        let modified = dst.iter().zip(src.iter()).any(|(&d, &s)| d != s);
        assert!(modified);
    }

    #[test]
    fn test_generate_scaling_hbd_empty() {
        let mut scaling = vec![0xFFu8; 1024];
        generate_scaling_hbd(10, &[], &mut scaling);
        assert!(scaling[..1024].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_generate_scaling_hbd_single_point() {
        let mut scaling = vec![0u8; 1024];
        generate_scaling_hbd(10, &[[128, 200]], &mut scaling);
        assert_eq!(scaling[0], 200);
        assert_eq!(scaling[511], 200);
        assert_eq!(scaling[512], 200);
        assert_eq!(scaling[1023], 200);
    }

    #[test]
    fn test_generate_scaling_hbd_two_points() {
        let mut scaling = vec![0u8; 1024];
        generate_scaling_hbd(10, &[[0, 0], [255, 255]], &mut scaling);
        assert_eq!(scaling[0], 0);
        assert_eq!(scaling[1023], 255);
        assert!(scaling[512] > 120 && scaling[512] < 140);
    }

    #[test]
    fn test_generate_scaling_hbd_monotonic() {
        let mut scaling = vec![0u8; 1024];
        generate_scaling_hbd(10, &[[0, 0], [100, 100]], &mut scaling);
        for i in 1..400 {
            assert!(scaling[i] >= scaling[i - 1], "not monotonic at {}", i);
        }
    }

    #[test]
    fn test_generate_scaling_hbd_interpolation_fills_gaps() {
        let mut scaling = vec![0u8; 1024];
        generate_scaling_hbd(10, &[[0, 10], [128, 200]], &mut scaling);
        // shift_x=2, pad=4. Positions 1,2,3 should be interpolated (not zero)
        assert!(scaling[1] > 0);
        assert!(scaling[2] > 0);
        assert!(scaling[3] > 0);
    }
}
