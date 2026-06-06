//! Bit-depth abstraction for the pixel/recon path.
//!
//! Replaces the C bitdepth templating pattern where `_tmpl.c` files are
//! compiled multiple times with different `BITDEPTH` defines. The 8-bit path
//! uses `u8` samples and a compile-time `bitdepth_max` of 255; the high-bit
//! depth path uses `u16` samples with a *runtime* `bitdepth_max` of
//! `(1 << bd) - 1` (1023 for 10-bit, 4095 for 12-bit), exactly mirroring
//! dav2d's `bitdepth.h`.

/// A storage type for one pixel sample (`u8` for 8bpc, `u16` for 10/12bpc).
pub trait Pixel: Copy + Default + Send + Sync + Into<i32> + 'static {
    /// Compile-time storage width in bits (8 for `u8`, 16 for `u16`). This is
    /// NOT the coded bit depth — for HBD the coded depth (10/12) is carried at
    /// runtime by [`BitDepth::bitdepth_max`].
    const BITDEPTH: u8;
    /// Largest value representable in the storage type.
    const MAX: Self;

    fn from_i32(v: i32) -> Self;
    fn as_u16(self) -> u16;
}

impl Pixel for u8 {
    const BITDEPTH: u8 = 8;
    const MAX: u8 = 0xFF;

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self as u16
    }
}

impl Pixel for u16 {
    const BITDEPTH: u8 = 16;
    const MAX: u16 = 0xFFFF;

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self
    }
}

/// Bit-depth dispatch trait, mirroring dav2d's `include/common/bitdepth.h`.
///
/// `BitDepth8` is a zero-sized type (the coded depth is always 8). `BitDepth16`
/// carries the coded bit depth at runtime (`bitdepth_max = (1 << bd) - 1`),
/// because a single `u16` storage type backs both 10- and 12-bit streams.
///
/// DSP kernels and the recon path are written once against this trait and
/// instantiated for both `BitDepth8` and `BitDepth16`; the `u8` instantiation
/// is byte-identical to the prior hard-coded `_8bpc` code.
pub trait BitDepth: Clone + Copy + Send + Sync + 'static {
    /// Sample storage type (`u8` or `u16`).
    type Pixel: Pixel;
    /// Intermediate/coefficient signed type wide enough for this depth.
    /// (`i16` for 8bpc, `i32` for HBD — matches dav2d's `coef`/`itxfm` ranges.)
    type Coef: Copy;

    /// Compile-time storage width in bits (8 or 16).
    const BPC: u8;

    /// Construct from the coded bit depth (8, 10 or 12). For `BitDepth8` the
    /// argument is ignored.
    fn new(bitdepth: u8) -> Self;

    /// Coded bit depth (8/10/12).
    fn bitdepth(&self) -> u8;

    /// `(1 << bitdepth) - 1` — the clip ceiling for reconstructed pixels.
    fn bitdepth_max(&self) -> i32;

    /// `bitdepth - 8` — the extra-precision shift used throughout dav2d's HBD
    /// kernels (deblock thresholds, cdef clips, intermediate rounding).
    #[inline(always)]
    fn bitdepth_min_8(&self) -> i32 {
        self.bitdepth() as i32 - 8
    }

    /// Clip a reconstructed sample into `[0, bitdepth_max]`.
    #[inline(always)]
    fn pixel_clip(&self, v: i32) -> Self::Pixel {
        let max = self.bitdepth_max();
        Self::Pixel::from_i32(v.clamp(0, max))
    }
}

/// 8-bit reconstruction (`u8` samples, fixed `bitdepth_max = 255`).
#[derive(Clone, Copy, Default)]
pub struct BitDepth8;

impl BitDepth for BitDepth8 {
    type Pixel = u8;
    type Coef = i16;
    const BPC: u8 = 8;

    #[inline(always)]
    fn new(_bitdepth: u8) -> Self {
        BitDepth8
    }
    #[inline(always)]
    fn bitdepth(&self) -> u8 {
        8
    }
    #[inline(always)]
    fn bitdepth_max(&self) -> i32 {
        255
    }
}

/// High-bit-depth reconstruction (`u16` samples, runtime 10/12-bit `bitdepth`).
#[derive(Clone, Copy)]
pub struct BitDepth16 {
    bitdepth: u8,
}

impl BitDepth for BitDepth16 {
    type Pixel = u16;
    type Coef = i32;
    const BPC: u8 = 16;

    #[inline(always)]
    fn new(bitdepth: u8) -> Self {
        debug_assert!(bitdepth == 10 || bitdepth == 12);
        BitDepth16 { bitdepth }
    }
    #[inline(always)]
    fn bitdepth(&self) -> u8 {
        self.bitdepth
    }
    #[inline(always)]
    fn bitdepth_max(&self) -> i32 {
        (1 << self.bitdepth) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitdepth8_constants() {
        let bd = BitDepth8::new(8);
        assert_eq!(bd.bitdepth(), 8);
        assert_eq!(bd.bitdepth_max(), 255);
        assert_eq!(bd.bitdepth_min_8(), 0);
        assert_eq!(bd.pixel_clip(-5), 0u8);
        assert_eq!(bd.pixel_clip(300), 255u8);
        assert_eq!(bd.pixel_clip(128), 128u8);
    }

    #[test]
    fn bitdepth16_10bit() {
        let bd = BitDepth16::new(10);
        assert_eq!(bd.bitdepth(), 10);
        assert_eq!(bd.bitdepth_max(), 1023);
        assert_eq!(bd.bitdepth_min_8(), 2);
        assert_eq!(bd.pixel_clip(-5), 0u16);
        assert_eq!(bd.pixel_clip(2000), 1023u16);
        assert_eq!(bd.pixel_clip(512), 512u16);
    }

    #[test]
    fn bitdepth16_12bit() {
        let bd = BitDepth16::new(12);
        assert_eq!(bd.bitdepth_max(), 4095);
        assert_eq!(bd.bitdepth_min_8(), 4);
        assert_eq!(bd.pixel_clip(9000), 4095u16);
    }

    #[test]
    fn pixel_trait_roundtrip() {
        assert_eq!(<u8 as Pixel>::from_i32(200), 200u8);
        assert_eq!(<u16 as Pixel>::from_i32(1000), 1000u16);
        assert_eq!(255u8.as_u16(), 255u16);
        assert_eq!(1023u16.as_u16(), 1023u16);
    }
}
