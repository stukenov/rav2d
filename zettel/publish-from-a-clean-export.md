# Publish from a clean export of the tag, not from the working tree

On 1 November 2026 another session was editing this checkout at the same
time (an `encoder-internals` feature for rav2e) and later reverted it.
`cargo publish` packages the working tree, so publishing from a shared
checkout can ship someone's uncommitted change. The dry run and the real
publish run on a worker from `git archive <tag>`; that also proves the
crate builds without the dav2d submodule.
