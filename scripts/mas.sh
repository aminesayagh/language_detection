#!/bin/bash

echo "Generating documentation Config..."

mkdir -p docs

# Generate tree structure with specific ignores
TREE_OUTPUT=$(tree -a -I 'node_modules|.git|.next|dist|.turbo|.cache|.vercel|coverage' \
     --dirsfirst \
     --charset=ascii)

{
  echo "# Project Tree Structure"
  echo "\`\`\`plaintext"
  echo "$TREE_OUTPUT"
  echo "\`\`\`"
} > docs/doc-tree.md

cw doc \
    --pattern "." \
    --output "docs/doc.md" \
    --compress false

echo "Documentation generated successfully."