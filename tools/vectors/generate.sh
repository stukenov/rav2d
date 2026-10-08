#!/bin/sh
# Encode the vectors listed in vectors.txt with avmenc and record avmdec's md5
# for each, so the clip and its reference output come from the AOM codec and
# never from the decoders under test.
#
# Usage: generate.sh <avm build dir> <bus_cif.y4m> <out dir> [name...]
# The avm build must be from the av2-normative branch; its commit goes into
# <out dir>/SOURCE so a vector can be traced to the encoder that made it.
set -eu
AVM=$1 SRC=$2 OUT=$3
shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

grep -v '^#' "$HERE/vectors.txt" | grep -v '^ *$' | while read -r name size frames bitdepth layout opts; do
    if [ $# -gt 0 ] && ! echo " $* " | grep -q " $name "; then continue; fi
    python3 "$HERE/crop_y4m.py" "$SRC" "$TMP/$name.y4m" --size "$size" --offset 96x64 \
        --frames "$frames" --bitdepth "$bitdepth" --layout "$layout"
    # shellcheck disable=SC2086 # $opts is a list of options
    "$AVM/avmenc" --good --cpu-used=5 --end-usage=q --qp=150 --limit="$frames" \
        --obu $opts -o "$OUT/avmenc-$name.obu" "$TMP/$name.y4m" > "$TMP/$name.enc.log" 2>&1 \
        || { echo "$name: avmenc failed"; tail -5 "$TMP/$name.enc.log"; continue; }
    # --rawvideo: hash the samples as decoded, the way dav2d-test-data does
    # (gen-md5.py); without it avmdec hashes something else.
    "$AVM/avmdec" --md5 --rawvideo "$OUT/avmenc-$name.obu" 2>/dev/null | cut -d' ' -f1 > "$OUT/avmenc-$name.obu.md5"
    echo "$name: $(wc -c < "$OUT/avmenc-$name.obu") bytes, md5 $(cat "$OUT/avmenc-$name.obu.md5")"
done
git -C "$AVM/../avm" log -1 --format='avm %H (%ci) %s' > "$OUT/SOURCE" 2>/dev/null || true
