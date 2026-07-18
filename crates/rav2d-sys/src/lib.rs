#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]
// bindgen copies C header comments verbatim, so array notation like `arr[0]`
// becomes a bogus intra-doc link in the generated bindings. These are not real
// Rust doc links; silence them so `cargo doc` (and docs.rs) build cleanly.
#![allow(rustdoc::broken_intra_doc_links)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn struct_sizes_nonzero() {
        assert!(mem::size_of::<Dav2dSettings>() > 0);
    }
}
