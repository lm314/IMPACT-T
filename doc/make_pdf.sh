#!/bin/bash
# Make the PDF versions of the GPU-port documents from their Markdown sources:
#
#   doc/make_pdf.sh                 # doc/GPU_PORT.pdf, doc/GPU_PORT_DESIGN.pdf
#
# Needs pandoc (3.1 or newer) and typst, from PATH or given as PANDOC=... TYPST=...
# (both are on conda-forge: "mamba create -n docs -c conda-forge pandoc typst").
# Page settings are in doc/pdf-style.yaml.
set -euo pipefail
cd "$(dirname "$0")"
PANDOC=${PANDOC:-pandoc}
TYPST=${TYPST:-typst}
for f in GPU_PORT GPU_PORT_DESIGN; do
    "$PANDOC" "$f.md" -o "$f.pdf" --pdf-engine="$TYPST" --metadata-file=pdf-style.yaml
    echo "wrote doc/$f.pdf"
done
