#!/bin/bash

# Find all folders matching the pattern
folders=$(find emulator -type d -name 'checkpoints' -print | sed 's|/checkpoints||')

# Delete each folder
for folder in $folders; do
    echo "Deleting folder: $folder"
    rm -r "$folder"
done