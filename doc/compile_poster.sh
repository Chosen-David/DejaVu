#!/bin/bash

# Euro-Par Poster/Demo Paper Compilation Script (2 pages)
# Usage: ./compile_poster.sh

echo "Compiling Euro-Par poster/demo paper (2 pages)..."

# First pass: generate aux files
pdflatex -interaction=nonstopmode europar_poster.tex

# Generate bibliography
bibtex europar_poster

# Second pass: incorporate bibliography
pdflatex -interaction=nonstopmode europar_poster.tex

# Third pass: resolve references
pdflatex -interaction=nonstopmode europar_poster.tex

# Check page count
if [ -f europar_poster.pdf ]; then
    echo ""
    echo "Compilation complete!"
    echo "Checking page count..."

    # Use pdfinfo or other method to check pages
    if command -v pdfinfo &> /dev/null; then
        PAGES=$(pdfinfo europar_poster.pdf | grep Pages | awk '{print $2}')
        echo "Page count: $PAGES pages"

        if [ "$PAGES" -le 2 ]; then
            echo "✓ Page count is within poster/demo limit (≤2 pages)"
        else
            echo "⚠ Warning: Page count exceeds poster/demo limit (expected ≤2, got $PAGES)"
            echo "  Consider further content reduction"
        fi
    fi

    echo ""
    echo "Output: europar_poster.pdf"
    echo ""
    echo "Note: This is the poster/demo version (2 pages)"
    echo "For short paper (6-8 pages): compile europar_short_paper.tex"
    echo "For full paper (12-14 pages): compile europar_paper.tex"
fi
