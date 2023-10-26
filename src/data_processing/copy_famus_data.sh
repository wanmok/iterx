#!/bin/bash

# Define the source directory
src="/brtx/601-nvme1/svashis3/FAMuS/data/cross_doc_role_extraction/iterx_format"

# Define the root directory of the repository
repo_root="$(git rev-parse --show-toplevel)"

# Define the destination directory
dest="${repo_root}/resources/data/famus"

echo "Source directory: $src"
echo "Destination directory: $dest"

# Create the destination directory if it doesn't exist
mkdir -p "$dest"

echo "Copying files..."

# Copy all the contents from the source directory to the destination directory
cp -R "$src"/* "$dest"

# Run vocab and definitions script
python create_famus_vocab_defs.py