/// Trait abstracting over pixel bit depths (8-bit and 16-bit).
///
/// Replaces the C bitdepth templating pattern where `_tmpl.c` files
/// are compiled multiple times with different `BITDEPTH` defines.
pub trait Pixel: Copy + Default + Send + Sync + Into<i32> + 'static {
    const BITDEPTH: u8;
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
