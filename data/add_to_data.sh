#!/bin/sh

# Get data if appropriate
make data.tar.gz

# Unzip, add file(s), zip again
gzip -d data.tar.gz
tar rf data.tar $@
gzip data.tar

# Update MD5
md5sum data.tar.gz > data.tar.gz.md5

# Add to .gitignore
echo >> .gitignore
for i in $@; do
    echo /$i >> .gitignore
done

echo
echo Added $@. Now:
echo 1. Publish data.tar.gz
echo 2. Commit new .gitignore and data.tar.gz.md5
echo
