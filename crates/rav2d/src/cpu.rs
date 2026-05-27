use std::sync::atomic::{AtomicU32, Ordering};

static CPU_FLAGS: AtomicU32 = AtomicU32::new(0);
static CPU_FLAGS_MASK: AtomicU32 = AtomicU32::new(u32::MAX);

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    pub const CPU_FLAG_SSE2: u32 = 1 << 0;
    pub const CPU_FLAG_SSSE3: u32 = 1 << 1;
    pub const CPU_FLAG_SSE41: u32 = 1 << 2;
    pub const CPU_FLAG_AVX2: u32 = 1 << 3;
    pub const CPU_FLAG_AVX512ICL: u32 = 1 << 4;

    pub fn detect() -> u32 {
        let mut flags = 0u32;
        if is_x86_feature_detected!("sse2") {
            flags |= CPU_FLAG_SSE2;
        }
        if is_x86_feature_detected!("ssse3") {
            flags |= CPU_FLAG_SSSE3;
        }
        if is_x86_feature_detected!("sse4.1") {
            flags |= CPU_FLAG_SSE41;
        }
        if is_x86_feature_detected!("avx2") {
            flags |= CPU_FLAG_AVX2;
        }
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
        {
            flags |= CPU_FLAG_AVX512ICL;
        }
        flags
    }
}

#[cfg(target_arch = "aarch64")]
pub mod arm {
    pub const CPU_FLAG_NEON: u32 = 1 << 0;
    pub const CPU_FLAG_DOTPROD: u32 = 1 << 1;
    pub const CPU_FLAG_I8MM: u32 = 1 << 2;
    pub const CPU_FLAG_SVE: u32 = 1 << 3;
    pub const CPU_FLAG_SVE2: u32 = 1 << 4;

    pub fn detect() -> u32 {
        let mut flags = CPU_FLAG_NEON;
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            flags |= CPU_FLAG_DOTPROD;
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            flags |= CPU_FLAG_I8MM;
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            flags |= CPU_FLAG_SVE;
        }
        if std::arch::is_aarch64_feature_detected!("sve2") {
            flags |= CPU_FLAG_SVE2;
        }
        flags
    }
}

pub fn init_cpu() {
    #[cfg(target_arch = "x86_64")]
    {
        CPU_FLAGS.store(x86::detect(), Ordering::Relaxed);
    }
    #[cfg(target_arch = "aarch64")]
    {
        CPU_FLAGS.store(arm::detect(), Ordering::Relaxed);
    }
}

pub fn set_cpu_flags_mask(mask: u32) {
    CPU_FLAGS_MASK.store(mask, Ordering::Relaxed);
}

pub fn get_cpu_flags() -> u32 {
    CPU_FLAGS.load(Ordering::Relaxed) & CPU_FLAGS_MASK.load(Ordering::Relaxed)
}

pub fn num_logical_processors() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_cpu() {
        init_cpu();
        let flags = get_cpu_flags();
        #[cfg(target_arch = "x86_64")]
        assert!(flags & x86::CPU_FLAG_SSE2 != 0);
        #[cfg(target_arch = "aarch64")]
        assert!(flags & arm::CPU_FLAG_NEON != 0);
        let _ = flags;
    }

    #[test]
    fn test_mask() {
        init_cpu();
        let all = get_cpu_flags();
        set_cpu_flags_mask(0);
        assert_eq!(get_cpu_flags(), 0);
        set_cpu_flags_mask(u32::MAX);
        assert_eq!(get_cpu_flags(), all);
    }

    #[test]
    fn test_num_logical_processors() {
        assert!(num_logical_processors() >= 1);
    }
}
