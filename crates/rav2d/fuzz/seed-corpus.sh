#!/bin/bash
# Seed a fuzz target's corpus from the conformance vectors.
#
# Starting from valid bitstreams is what makes a format fuzzer productive: the
# mutator reaches deep decoder state in a few flips instead of spending its
# budget rediscovering the OBU header. Existing corpus entries are left alone,
# so this is safe to run over a corpus restored from cache.
#
# Usage: seed-corpus.sh <target-name>   (run from anywhere inside the repo)
set -euo pipefail

target=${1:?usage: seed-corpus.sh <target-name>}
fuzz_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$fuzz_dir/../../.." && pwd)"
corpus="$fuzz_dir/corpus/$target"

mkdir -p "$corpus"

# `decode_settings` reads the first two bytes as a decoder configuration, so a
# seed has to carry a prefix or the stream starts two bytes in.
prefix=0
if [ "$target" = "decode_settings" ]; then
  prefix=2
fi

seeded=0
for vector in "$repo_root"/dav2d/media/*.obu "$repo_root"/crates/rav2d/tests/data/*.obu \
              "$repo_root"/crates/rav2d/tests/data/fuzz-regressions/*; do
  [ -f "$vector" ] || continue
  dest="$corpus/seed-$(basename "$vector")"
  [ -e "$dest" ] && continue
  if [ "$prefix" -eq 0 ]; then
    cp "$vector" "$dest"
  else
    # A zero prefix means default-ish settings; the mutator explores the rest.
    { head -c "$prefix" /dev/zero; cat "$vector"; } > "$dest"
  fi
  seeded=$((seeded + 1))
done

echo "seeded $seeded new file(s) into $corpus ($(ls -1 "$corpus" | wc -l | tr -d ' ') total)"
