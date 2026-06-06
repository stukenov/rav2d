use wide::i32x8;

use crate::levels::{N_TX_1D_TYPES, N_TX_SIZES};

pub type Itx1dFn = fn(c: &mut [i32], stride: usize);

static DCT8_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];

static DCT16_KERNEL: [i8; 64] = [
    90, 87, 80, 70, 57, 43, 26, 9, 87, 57, 9, -43, -80, -90, -70, -26, 80, 9, -70, -87, -26, 57,
    90, 43, 70, -43, -87, 9, 90, 26, -80, -57, 57, -80, -26, 90, -9, -87, 43, 70, 43, -90, 57, 26,
    -87, 70, 9, -80, 26, -70, 90, -80, 43, 9, -57, 87, 9, -26, 43, -57, 70, -80, 87, -90,
];

static DCT32_KERNEL: [i8; 256] = [
    90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4, 90, 82, 67, 47, 22, -4, -30,
    -54, -73, -85, -90, -88, -78, -61, -39, -13, 88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39,
    73, 90, 85, 61, 22, 85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30,
    82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67, 4, 73, 88, 39, 78, -4, -82, -73, 13, 85,
    67, -22, -88, -61, 30, 90, 54, -39, -90, -47, 73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61,
    -47, -88, -4, 85, 54, 67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47, -73, -61,
    61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67, 54, -85, -4, 88, -47,
    -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73, 47, -90, 39, 54, -90, 30, 61, -88, 22, 67,
    -85, 13, 73, -82, 4, 78, 39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82,
    30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85, 22, -61, 85, -90, 73,
    -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88, 13, -39, 61, -78, 88, -90, 85, -73, 54, -30,
    4, 22, -47, 67, -82, 90, 4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
];

static ADST4_KERNEL: [i8; 16] = [
    18, 50, 75, 89, 50, 89, 18, -75, 75, 18, -89, 50, 89, -75, 50, -18,
];

static ADST8_KERNEL: [i8; 64] = [
    11, 34, 54, 71, 84, 88, 79, 50, 28, 74, 89, 68, 17, -44, -83, -69, 44, 89, 48, -41, -89, -44,
    50, 81, 58, 76, -34, -86, 10, 88, 6, -84, 70, 39, -87, 1, 86, -44, -59, 78, 79, -12, -66, 87,
    -35, -44, 86, -62, 86, -58, 12, 38, -75, 88, -74, 40, 89, -86, 79, -70, 58, -44, 29, -14,
];

static ADST16_KERNEL: [i8; 256] = [
    8, 25, 41, 55, 67, 77, 84, 88, 89, 87, 81, 73, 62, 48, 33, 17, 17, 48, 73, 87, 88, 77, 55, 25,
    -8, -41, -67, -84, -89, -81, -62, -33, 25, 67, 88, 81, 48, 0, -48, -81, -88, -67, -25, 25, 67,
    88, 81, 48, 33, 81, 84, 41, -25, -77, -87, -48, 17, 73, 88, 55, -8, -67, -89, -62, 41, 88, 62,
    -17, -81, -77, -8, 67, 87, 33, -48, -89, -55, 25, 84, 73, 48, 88, 25, -67, -81, 0, 81, 67, -25,
    -88, -48, 48, 88, 25, -67, -81, 55, 81, -17, -89, -25, 77, 62, -48, -84, 8, 88, 33, -73, -67,
    41, 87, 62, 67, -55, -73, 48, 77, -41, -81, 33, 84, -25, -87, 17, 88, -8, -89, 67, 48, -81,
    -25, 88, 0, -88, 25, 81, -48, -67, 67, 48, -81, -25, 88, 73, 25, -89, 33, 67, -77, -17, 88,
    -41, -62, 81, 8, -87, 48, 55, -84, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77,
    77, 81, -25, -48, 88, -67, 0, 67, -88, 48, 25, -81, 81, -25, -48, 88, -67, 84, -48, -8, 62,
    -88, 77, -33, -25, 73, -89, 67, -17, -41, 81, -87, 55, 87, -67, 33, 8, -48, 77, -89, 81, -55,
    17, 25, -62, 84, -88, 73, -41, 88, -81, 67, -48, 25, 0, -25, 48, -67, 81, -88, 88, -81, 67,
    -48, 25, 89, -88, 87, -84, 81, -77, 73, -67, 62, -55, 48, -41, 33, -25, 17, -8,
];

static FLIPADST4_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];

static FLIPADST16_KERNEL: [i8; 256] = [
    89, 88, 87, 84, 81, 77, 73, 67, 62, 55, 48, 41, 33, 25, 17, 8, 88, 81, 67, 48, 25, 0, -25, -48,
    -67, -81, -88, -88, -81, -67, -48, -25, 87, 67, 33, -8, -48, -77, -89, -81, -55, -17, 25, 62,
    84, 88, 73, 41, 84, 48, -8, -62, -88, -77, -33, 25, 73, 89, 67, 17, -41, -81, -87, -55, 81, 25,
    -48, -88, -67, 0, 67, 88, 48, -25, -81, -81, -25, 48, 88, 67, 77, 0, -77, -77, 0, 77, 77, 0,
    -77, -77, 0, 77, 77, 0, -77, -77, 73, -25, -89, -33, 67, 77, -17, -88, -41, 62, 81, -8, -87,
    -48, 55, 84, 67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67, -67, 48, 81, -25, -88, 62, -67,
    -55, 73, 48, -77, -41, 81, 33, -84, -25, 87, 17, -88, -8, 89, 55, -81, -17, 89, -25, -77, 62,
    48, -84, -8, 88, -33, -73, 67, 41, -87, 48, -88, 25, 67, -81, 0, 81, -67, -25, 88, -48, -48,
    88, -25, -67, 81, 41, -88, 62, 17, -81, 77, -8, -67, 87, -33, -48, 89, -55, -25, 84, -73, 33,
    -81, 84, -41, -25, 77, -87, 48, 17, -73, 88, -55, -8, 67, -89, 62, 25, -67, 88, -81, 48, 0,
    -48, 81, -88, 67, -25, -25, 67, -88, 81, -48, 17, -48, 73, -87, 88, -77, 55, -25, -8, 41, -67,
    84, -89, 81, -62, 33, 8, -25, 41, -55, 67, -77, 84, -88, 89, -87, 81, -73, 62, -48, 33, -17,
];

static DDT8_KERNEL: [i8; 64] = [
    4, 6, 22, 57, 96, 103, 78, 56, 7, 14, 48, 94, 73, -17, -79, -96, 15, 36, 85, 76, -43, -80, 7,
    98, 33, 77, 88, -26, -69, 56, 56, -77, 65, 100, 0, -73, 55, 15, -82, 54, 98, 45, -86, 34, 20,
    -66, 79, -33, 106, -57, -23, 54, -71, 75, -56, 19, 80, -98, 82, -66, 53, -41, 26, -6,
];

static DDT16_KERNEL: [i8; 256] = [
    12, 17, 37, 45, 47, 60, 64, 82, 89, 100, 92, 84, 69, 50, 51, 44, 15, 23, 49, 60, 60, 74, 70,
    73, 48, 9, -35, -71, -83, -79, -89, -95, 19, 30, 60, 69, 61, 64, 40, 3, -53, -99, -91, -46, 2,
    47, 73, 124, 23, 38, 69, 73, 49, 28, -19, -80, -96, -45, 42, 88, 75, 14, -17, -126, 30, 48, 75,
    66, 19, -31, -79, -91, -5, 84, 71, -16, -78, -60, -45, 108, 39, 61, 75, 40, -29, -87, -78, 10,
    89, 36, -69, -67, 18, 67, 89, -81, 51, 76, 61, -8, -77, -82, 11, 94, 16, -81, -22, 79, 50, -37,
    -103, 54, 66, 87, 29, -65, -83, 4, 92, 18, -83, 4, 85, -22, -85, -6, 97, -30, 78, 83, -18, -91,
    -16, 88, 28, -84, 12, 73, -60, -46, 81, 49, -83, 16, 88, 59, -67, -57, 75, 54, -85, -5, 75,
    -60, -17, 84, -43, -80, 71, -6, 94, 19, -96, 21, 93, -55, -41, 80, -51, -17, 77, -68, -6, 98,
    -56, 1, 97, -30, -83, 86, 3, -77, 82, -17, -43, 76, -70, 15, 53, -99, 44, 3, 93, -73, -28, 81,
    -92, 29, 39, -70, 81, -55, 11, 46, -81, 90, -31, -4, 83, -99, 40, 8, -74, 88, -83, 47, -14,
    -21, 56, -83, 88, -71, 22, 5, 68, -99, 84, -69, 32, 3, -37, 55, -75, 81, -83, 82, -69, 48, -11,
    -3, 50, -76, 83, -90, 97, -86, 83, -68, 67, -56, 49, -40, 32, -19, 5, 2,
];

#[inline(never)]
fn inv_dct_1d(c: &mut [i32], stride: usize, mat: &[i8], n: usize) {
    let mut a = [0i32; 16];
    let mut b = [0i32; 16];
    let k = n * 2 - 1;
    let mut mi = 0;

    for i in 0..n {
        let mut sum = 0i32;
        let mut j = 1;
        while j <= k {
            sum += mat[mi] as i32 * c[j * stride];
            mi += 1;
            j += 2;
        }
        a[i] = c[i * 2 * stride];
        b[i] = sum;
    }

    for i in 0..n {
        c[i * stride] = a[i] + b[i];
        c[(k - i) * stride] = a[i] - b[i];
    }
}

#[inline(never)]
fn inv_dct4_1d(c: &mut [i32], stride: usize) {
    let a0 = c[0 * stride] * 64 + c[2 * stride] * 64;
    let a1 = c[0 * stride] * 64 - c[2 * stride] * 64;
    let b0 = c[stride] * 83 + c[3 * stride] * 35;
    let b1 = c[stride] * 35 - c[3 * stride] * 83;

    c[0 * stride] = a0 + b0;
    c[stride] = a1 + b1;
    c[2 * stride] = a1 - b1;
    c[3 * stride] = a0 - b0;
}

#[inline(never)]
fn inv_dct8_1d(c: &mut [i32], stride: usize) {
    inv_dct4_1d(c, 2 * stride);
    inv_dct_1d(c, stride, &DCT8_KERNEL, 4);
}

#[inline(never)]
fn inv_dct16_1d(c: &mut [i32], stride: usize) {
    inv_dct8_1d(c, 2 * stride);
    inv_dct_1d(c, stride, &DCT16_KERNEL, 8);
}

fn inv_dct32_1d(c: &mut [i32], stride: usize) {
    inv_dct16_1d(c, 2 * stride);
    inv_dct_1d(c, stride, &DCT32_KERNEL, 16);
}

#[inline(never)]
fn inv_dst_1d(c: &mut [i32], start: usize, stride: usize, mat: &[i8], n: usize, flip: bool) {
    let mut sums = [0i32; 16];
    let mut mi = 0;

    for i in 0..n {
        let mut sum = 0i32;
        for j in 0..n {
            sum += mat[mi] as i32 * c[start + j * stride];
            mi += 1;
        }
        sums[i] = sum;
    }

    if flip {
        for i in 0..n {
            c[start + (n - 1 - i) * stride] = sums[i];
        }
    } else {
        for i in 0..n {
            c[start + i * stride] = sums[i];
        }
    }
}

fn inv_adst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST4_KERNEL, 4, false);
}

fn inv_adst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST8_KERNEL, 8, false);
}

fn inv_adst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST16_KERNEL, 16, false);
}

fn inv_flipadst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &FLIPADST4_KERNEL, 4, false);
}

fn inv_flipadst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST8_KERNEL, 8, true);
}

fn inv_flipadst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &FLIPADST16_KERNEL, 16, false);
}

fn inv_ddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT8_KERNEL, 8, false);
}

fn inv_ddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT16_KERNEL, 16, false);
}

fn inv_flipddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT8_KERNEL, 8, true);
}

fn inv_flipddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT16_KERNEL, 16, true);
}

fn inv_identity4_1d(c: &mut [i32], stride: usize) {
    for i in 0..4 {
        c[stride * i] *= 128;
    }
}

fn inv_identity8_1d(c: &mut [i32], stride: usize) {
    for i in 0..8 {
        c[stride * i] *= 181;
    }
}

fn inv_identity16_1d(c: &mut [i32], stride: usize) {
    for i in 0..16 {
        c[stride * i] *= 256;
    }
}

fn inv_identity32_1d(c: &mut [i32], stride: usize) {
    for i in 0..32 {
        c[stride * i] *= 362;
    }
}

// ---------------------------------------------------------------------------
// SoA-batched (`x8`) inverse transforms — used by `inv_txfm_add`'s second pass
// (the stride-`sw` column transform). Eight adjacent columns are processed per
// call as an `i32x8` lane vector: column `x..x+8` at transform position `p` is
// the contiguous slice `c[base + p*stride .. +8]`, so every butterfly operand
// is one contiguous load. The math mirrors the scalar transforms above exactly
// (pure i32 arithmetic, no rounding), so the result is bit-identical; the unit
// tests assert this against the scalar fns column-by-column.

/// `(&mut [i32], base, stride)` — vectorized 1-D transform over 8 columns.
pub type Itx1dFnX8 = fn(&mut [i32], usize, usize);

#[inline(always)]
fn ldx8(c: &[i32], off: usize) -> i32x8 {
    i32x8::from([
        c[off],
        c[off + 1],
        c[off + 2],
        c[off + 3],
        c[off + 4],
        c[off + 5],
        c[off + 6],
        c[off + 7],
    ])
}

#[inline(always)]
fn stx8(c: &mut [i32], off: usize, v: i32x8) {
    c[off..off + 8].copy_from_slice(&v.to_array());
}

#[inline(always)]
fn mulc(v: i32x8, k: i32) -> i32x8 {
    v * i32x8::splat(k)
}

fn inv_dct_1d_x8(c: &mut [i32], base: usize, stride: usize, mat: &[i8], n: usize) {
    let zero = i32x8::splat(0);
    let mut a = [zero; 16];
    let mut b = [zero; 16];
    let k = n * 2 - 1;
    let mut mi = 0;
    for i in 0..n {
        let mut sum = zero;
        let mut j = 1;
        while j <= k {
            sum += mulc(ldx8(c, base + j * stride), mat[mi] as i32);
            mi += 1;
            j += 2;
        }
        a[i] = ldx8(c, base + i * 2 * stride);
        b[i] = sum;
    }
    for i in 0..n {
        stx8(c, base + i * stride, a[i] + b[i]);
        stx8(c, base + (k - i) * stride, a[i] - b[i]);
    }
}

fn inv_dct4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let c0 = ldx8(c, base);
    let c1 = ldx8(c, base + stride);
    let c2 = ldx8(c, base + 2 * stride);
    let c3 = ldx8(c, base + 3 * stride);
    let a0 = mulc(c0, 64) + mulc(c2, 64);
    let a1 = mulc(c0, 64) - mulc(c2, 64);
    let b0 = mulc(c1, 83) + mulc(c3, 35);
    let b1 = mulc(c1, 35) - mulc(c3, 83);
    stx8(c, base, a0 + b0);
    stx8(c, base + stride, a1 + b1);
    stx8(c, base + 2 * stride, a1 - b1);
    stx8(c, base + 3 * stride, a0 - b0);
}

fn inv_dct8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dct4_1d_x8(c, base, 2 * stride);
    inv_dct_1d_x8(c, base, stride, &DCT8_KERNEL, 4);
}

fn inv_dct16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dct8_1d_x8(c, base, 2 * stride);
    inv_dct_1d_x8(c, base, stride, &DCT16_KERNEL, 8);
}

fn inv_dct32_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dct16_1d_x8(c, base, 2 * stride);
    inv_dct_1d_x8(c, base, stride, &DCT32_KERNEL, 16);
}

fn inv_dst_1d_x8(c: &mut [i32], base: usize, stride: usize, mat: &[i8], n: usize, flip: bool) {
    let zero = i32x8::splat(0);
    let mut sums = [zero; 16];
    let mut mi = 0;
    for sum in sums.iter_mut().take(n) {
        let mut acc = zero;
        for j in 0..n {
            acc += mulc(ldx8(c, base + j * stride), mat[mi] as i32);
            mi += 1;
        }
        *sum = acc;
    }
    if flip {
        for i in 0..n {
            stx8(c, base + (n - 1 - i) * stride, sums[i]);
        }
    } else {
        for i in 0..n {
            stx8(c, base + i * stride, sums[i]);
        }
    }
}

fn inv_adst4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST4_KERNEL, 4, false);
}
fn inv_adst8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST8_KERNEL, 8, false);
}
fn inv_adst16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST16_KERNEL, 16, false);
}
fn inv_flipadst4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &FLIPADST4_KERNEL, 4, false);
}
fn inv_flipadst8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST8_KERNEL, 8, true);
}
fn inv_flipadst16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &FLIPADST16_KERNEL, 16, false);
}
fn inv_ddt8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT8_KERNEL, 8, false);
}
fn inv_ddt16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT16_KERNEL, 16, false);
}
fn inv_flipddt8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT8_KERNEL, 8, true);
}
fn inv_flipddt16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT16_KERNEL, 16, true);
}

fn inv_identity4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    for i in 0..4 {
        let off = base + stride * i;
        stx8(c, off, mulc(ldx8(c, off), 128));
    }
}
fn inv_identity8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    for i in 0..8 {
        let off = base + stride * i;
        stx8(c, off, mulc(ldx8(c, off), 181));
    }
}
fn inv_identity16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    for i in 0..16 {
        let off = base + stride * i;
        stx8(c, off, mulc(ldx8(c, off), 256));
    }
}
fn inv_identity32_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    for i in 0..32 {
        let off = base + stride * i;
        stx8(c, off, mulc(ldx8(c, off), 362));
    }
}

/// SoA-batched counterpart of [`TX1D_FNS`] (same `[tx_size][tx_1d_type]` layout).
pub static TX1D_FNS_X8: [[Option<Itx1dFnX8>; N_TX_1D_TYPES - 1]; N_TX_SIZES] = {
    const DCT: usize = 0;
    const IDENTITY: usize = 1;
    const ADST: usize = 2;
    const FLIPADST: usize = 3;
    const DDT: usize = 4;
    const FLIPDDT: usize = 5;
    const NONE: Option<Itx1dFnX8> = None;

    let mut t = [[NONE; N_TX_1D_TYPES - 1]; N_TX_SIZES];

    t[0][DCT] = Some(inv_dct4_1d_x8 as Itx1dFnX8);
    t[0][IDENTITY] = Some(inv_identity4_1d_x8);
    t[0][ADST] = Some(inv_adst4_1d_x8);
    t[0][FLIPADST] = Some(inv_flipadst4_1d_x8);

    t[1][DCT] = Some(inv_dct8_1d_x8);
    t[1][IDENTITY] = Some(inv_identity8_1d_x8);
    t[1][ADST] = Some(inv_adst8_1d_x8);
    t[1][FLIPADST] = Some(inv_flipadst8_1d_x8);
    t[1][DDT] = Some(inv_ddt8_1d_x8);
    t[1][FLIPDDT] = Some(inv_flipddt8_1d_x8);

    t[2][DCT] = Some(inv_dct16_1d_x8);
    t[2][IDENTITY] = Some(inv_identity16_1d_x8);
    t[2][ADST] = Some(inv_adst16_1d_x8);
    t[2][FLIPADST] = Some(inv_flipadst16_1d_x8);
    t[2][DDT] = Some(inv_ddt16_1d_x8);
    t[2][FLIPDDT] = Some(inv_flipddt16_1d_x8);

    t[3][DCT] = Some(inv_dct32_1d_x8);
    t[3][IDENTITY] = Some(inv_identity32_1d_x8);

    t[4][DCT] = Some(inv_dct32_1d_x8);

    t
};

pub fn inv_wht4_1d(c: &mut [i32], stride: usize) {
    let in0 = c[0 * stride];
    let in1 = c[stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let t0 = in0 + in1;
    let t2 = in2 - in3;
    let t4 = (t0 - t2) >> 1;
    let t3 = t4 - in3;
    let t1 = t4 - in1;

    c[0 * stride] = t0 - t3;
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = t2 + t1;
}

pub fn cctx(u: &mut [i32], v: &mut [i32], angle: &[i16; 3], sz: usize, bitdepth: i32) {
    debug_assert!(sz.is_power_of_two() && (16..=1024).contains(&sz));
    let min = -(1 << (bitdepth + 7));
    let max = (1 << (bitdepth + 7)) - 1;
    let sina = angle[0] as i32;
    let cosa = angle[1] as i32;
    debug_assert!(angle[2] == -angle[0]);
    crate::simd::cctx_row(u, v, sina, cosa, sz, min, max);
}

pub fn inv_wht_wht_4x4(coeff: &[i32; 16], tmp: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            tmp[y * 4 + x] = coeff[y + x * 4] >> 3;
        }
        inv_wht4_1d(&mut tmp[y * 4..], 1);
    }
    for x in 0..4 {
        inv_wht4_1d(&mut tmp[x..], 4);
    }
}

// Tx1dType indices: Dct=0, Identity=1, Adst=2, FlipAdst=3, Ddt=4, FlipDdt=5
// Table excludes Wht (index 6), hence N_TX_1D_TYPES - 1 = 6 columns
pub static TX1D_FNS: [[Option<Itx1dFn>; N_TX_1D_TYPES - 1]; N_TX_SIZES] = {
    const DCT: usize = 0;
    const IDENTITY: usize = 1;
    const ADST: usize = 2;
    const FLIPADST: usize = 3;
    const DDT: usize = 4;
    const FLIPDDT: usize = 5;
    const NONE: Option<Itx1dFn> = None;

    let mut t = [[NONE; N_TX_1D_TYPES - 1]; N_TX_SIZES];

    // TX_4X4
    t[0][DCT] = Some(inv_dct4_1d);
    t[0][IDENTITY] = Some(inv_identity4_1d);
    t[0][ADST] = Some(inv_adst4_1d);
    t[0][FLIPADST] = Some(inv_flipadst4_1d);

    // TX_8X8
    t[1][DCT] = Some(inv_dct8_1d);
    t[1][IDENTITY] = Some(inv_identity8_1d);
    t[1][ADST] = Some(inv_adst8_1d);
    t[1][FLIPADST] = Some(inv_flipadst8_1d);
    t[1][DDT] = Some(inv_ddt8_1d);
    t[1][FLIPDDT] = Some(inv_flipddt8_1d);

    // TX_16X16
    t[2][DCT] = Some(inv_dct16_1d);
    t[2][IDENTITY] = Some(inv_identity16_1d);
    t[2][ADST] = Some(inv_adst16_1d);
    t[2][FLIPADST] = Some(inv_flipadst16_1d);
    t[2][DDT] = Some(inv_ddt16_1d);
    t[2][FLIPDDT] = Some(inv_flipddt16_1d);

    // TX_32X32
    t[3][DCT] = Some(inv_dct32_1d);
    t[3][IDENTITY] = Some(inv_identity32_1d);

    // TX_64X64
    t[4][DCT] = Some(inv_dct32_1d);

    t
};

/// Generic residual add (`residual_add` in `itx_tmpl.c`). `dst` holds samples of
/// type `BD::Pixel`; the reconstructed value is clipped into `[0, bitdepth_max]`.
pub fn residual_add<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    c: &[i32],
    w: usize,
    h: usize,
    rnd: i32,
    shift: i32,
    dpcm_flag: u8,
) {
    match dpcm_flag {
        1 => {
            let mut ci = 0;
            for y in 0..h {
                let mut acc = 0i32;
                for x in 0..w {
                    acc += (c[ci] + rnd) >> shift;
                    let p = dst[y * stride + x].into();
                    dst[y * stride + x] = bd.pixel_clip(p + acc);
                    ci += 1;
                }
            }
        }
        2 => {
            for x in 0..w {
                let mut acc = 0i32;
                for y in 0..h {
                    acc += (c[y * w + x] + rnd) >> shift;
                    let p = dst[y * stride + x].into();
                    dst[y * stride + x] = bd.pixel_clip(p + acc);
                }
            }
        }
        // dpcm_flag 0 — and any non-1/2 value, which is an invalid combination
        // the C reference reaches only with asserts disabled: itx_tmpl.c's
        // `switch (dpcm_flag) { default: assert(0); case 0: ... }` falls through
        // from `default` into `case 0`, i.e. the plain non-DPCM residual add.
        _ => {
            for y in 0..h {
                let row = y * stride;
                if row >= dst.len() {
                    break;
                }
                let cw = y * w;
                let d = &mut dst[row..];
                let cr = &c[cw.min(c.len())..];
                let n = w.min(d.len()).min(cr.len());
                crate::simd::residual_add_row(bd, d, cr, n, rnd, shift);
            }
        }
    }
}

/// 8bpc residual add — byte-identical to the prior hand-written kernel.
#[inline]
pub fn residual_add_8bpc(
    dst: &mut [u8],
    stride: usize,
    c: &[i32],
    w: usize,
    h: usize,
    rnd: i32,
    shift: i32,
    dpcm_flag: u8,
) {
    residual_add(
        crate::pixel::BitDepth8,
        dst,
        stride,
        c,
        w,
        h,
        rnd,
        shift,
        dpcm_flag,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tx1d_fns_table_size() {
        assert_eq!(TX1D_FNS.len(), N_TX_SIZES);
        assert_eq!(TX1D_FNS[0].len(), N_TX_1D_TYPES - 1);
    }

    #[test]
    fn test_tx1d_fns_populated() {
        assert!(TX1D_FNS[0][0].is_some()); // TX_4X4 DCT
        assert!(TX1D_FNS[1][0].is_some()); // TX_8X8 DCT
        assert!(TX1D_FNS[2][0].is_some()); // TX_16X16 DCT
        assert!(TX1D_FNS[3][0].is_some()); // TX_32X32 DCT
        assert!(TX1D_FNS[4][0].is_some()); // TX_64X64 DCT
        assert!(TX1D_FNS[3][2].is_none()); // TX_32X32 ADST = None
        assert!(TX1D_FNS[4][1].is_none()); // TX_64X64 Identity = None
    }

    #[test]
    fn x8_transforms_match_scalar() {
        // Each SoA `x8` transform must equal the scalar transform applied to
        // each of its 8 columns independently (this is exactly how the second
        // pass of `inv_txfm_add` invokes them).
        let pairs: &[(&str, Itx1dFn, Itx1dFnX8)] = &[
            ("dct4", inv_dct4_1d, inv_dct4_1d_x8),
            ("dct8", inv_dct8_1d, inv_dct8_1d_x8),
            ("dct16", inv_dct16_1d, inv_dct16_1d_x8),
            ("dct32", inv_dct32_1d, inv_dct32_1d_x8),
            ("id4", inv_identity4_1d, inv_identity4_1d_x8),
            ("id8", inv_identity8_1d, inv_identity8_1d_x8),
            ("id16", inv_identity16_1d, inv_identity16_1d_x8),
            ("id32", inv_identity32_1d, inv_identity32_1d_x8),
            ("adst4", inv_adst4_1d, inv_adst4_1d_x8),
            ("adst8", inv_adst8_1d, inv_adst8_1d_x8),
            ("adst16", inv_adst16_1d, inv_adst16_1d_x8),
            ("flipadst4", inv_flipadst4_1d, inv_flipadst4_1d_x8),
            ("flipadst8", inv_flipadst8_1d, inv_flipadst8_1d_x8),
            ("flipadst16", inv_flipadst16_1d, inv_flipadst16_1d_x8),
            ("ddt8", inv_ddt8_1d, inv_ddt8_1d_x8),
            ("ddt16", inv_ddt16_1d, inv_ddt16_1d_x8),
            ("flipddt8", inv_flipddt8_1d, inv_flipddt8_1d_x8),
            ("flipddt16", inv_flipddt16_1d, inv_flipddt16_1d_x8),
        ];
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // ±~2^19, matching the post-row-clip intermediate range.
            (state as i32) >> 12
        };
        let stride = 8usize;
        let len = 64 * stride;
        for (name, scalar, x8) in pairs {
            let base: Vec<i32> = (0..len).map(|_| rng()).collect();
            let mut a = base.clone();
            x8(&mut a, 0, stride);
            let mut b = base.clone();
            for col in 0..8 {
                scalar(&mut b[col..], stride);
            }
            assert_eq!(a, b, "x8 mismatch in {name}");
        }
    }

    #[test]
    fn test_inv_dct4_1d_dc() {
        let mut c = [1024, 0, 0, 0];
        inv_dct4_1d(&mut c, 1);
        assert_eq!(c[0], 64 * 1024);
        assert_eq!(c[1], 64 * 1024);
        assert_eq!(c[2], 64 * 1024);
        assert_eq!(c[3], 64 * 1024);
    }

    #[test]
    fn test_inv_identity4_1d() {
        let mut c = [10, 20, 30, 40];
        inv_identity4_1d(&mut c, 1);
        assert_eq!(c, [1280, 2560, 3840, 5120]);
    }

    #[test]
    fn test_inv_wht4_1d() {
        let mut c = [100, 0, 0, 0];
        inv_wht4_1d(&mut c, 1);
        assert_eq!(c[0] + c[1] + c[2] + c[3], 200);
    }

    #[test]
    fn test_inv_dct8_1d_dc() {
        let mut c = [1000, 0, 0, 0, 0, 0, 0, 0];
        inv_dct8_1d(&mut c, 1);
        let expected = 64 * 1000;
        for &v in &c {
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn test_inv_adst4_symmetry() {
        let mut c1 = [100, 200, 300, 400];
        inv_adst4_1d(&mut c1, 1);
        let sum: i32 = c1.iter().sum();
        assert_ne!(sum, 0);
    }

    #[test]
    fn test_inv_wht_wht_4x4_zero() {
        let coeff = [0i32; 16];
        let mut tmp = [0i32; 16];
        inv_wht_wht_4x4(&coeff, &mut tmp);
        assert!(tmp.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_inv_wht_wht_4x4_dc() {
        let mut coeff = [0i32; 16];
        coeff[0] = 1024;
        let mut tmp = [0i32; 16];
        inv_wht_wht_4x4(&coeff, &mut tmp);
        assert!(tmp.iter().all(|&v| v == tmp[0]));
    }

    #[test]
    fn test_cctx_identity() {
        let mut u = [100i32; 16];
        let mut v = [200i32; 16];
        let angle: [i16; 3] = [0, 256, 0]; // cosa=256 (~1.0), sina=0
        cctx(&mut u, &mut v, &angle, 16, 8);
        assert_eq!(u[0], 100);
        assert_eq!(v[0], 200);
    }

    #[test]
    fn test_cctx_swap() {
        let mut u = [256i32; 16];
        let mut v = [0i32; 16];
        let angle: [i16; 3] = [256, 0, -256]; // cosa=0, sina=256 (~1.0)
        cctx(&mut u, &mut v, &angle, 16, 8);
        assert!(v[0] > 0);
    }

    #[test]
    fn test_cctx_clamp_8bpc() {
        let mut u = [30000i32; 16];
        let mut v = [30000i32; 16];
        let angle: [i16; 3] = [200, 200, -200];
        cctx(&mut u, &mut v, &angle, 16, 8);
        let max = (1 << 15) - 1;
        let min = -(1 << 15);
        for i in 0..16 {
            assert!(u[i] >= min && u[i] <= max);
            assert!(v[i] >= min && v[i] <= max);
        }
    }

    #[test]
    fn test_residual_add_no_dpcm() {
        let mut dst = vec![128u8; 16];
        let c = vec![10i32; 16];
        residual_add_8bpc(&mut dst, 4, &c, 4, 4, 0, 0, 0);
        for &v in &dst {
            assert_eq!(v, 138);
        }
    }

    #[test]
    fn test_residual_add_clamps() {
        let mut dst = vec![250u8; 4];
        let c = vec![100i32; 4];
        residual_add_8bpc(&mut dst, 4, &c, 4, 1, 0, 0, 0);
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_residual_add_dpcm_h() {
        let mut dst = vec![100u8; 4];
        let c = vec![5i32; 4];
        residual_add_8bpc(&mut dst, 4, &c, 4, 1, 0, 0, 1);
        assert_eq!(dst[0], 105);
        assert_eq!(dst[1], 110);
        assert_eq!(dst[2], 115);
        assert_eq!(dst[3], 120);
    }

    #[test]
    fn test_residual_add_dpcm_v() {
        let mut dst = vec![100u8; 8];
        let c = vec![5i32; 8];
        residual_add_8bpc(&mut dst, 2, &c, 2, 4, 0, 0, 2);
        assert_eq!(dst[0], 105);
        assert_eq!(dst[2], 110);
        assert_eq!(dst[4], 115);
        assert_eq!(dst[6], 120);
    }

    #[test]
    fn test_residual_add_with_shift() {
        let mut dst = vec![128u8; 4];
        let c = vec![64i32; 4];
        residual_add_8bpc(&mut dst, 4, &c, 4, 1, 32, 6, 0);
        for &v in &dst {
            assert!(v > 128);
        }
    }
}
