#!/bin/bash
# Script to free up space on GitHub Actions runner

# Best-effort cleanup — nothing here should block the build.
set +e

echo "Freeing up disk space on GitHub Actions runner..."
df -h

# Remove unnecessary large packages (best-effort, packages may not exist)
echo "Removing unnecessary large packages..."
sudo apt-get remove -y '^dotnet-.*' '^llvm-.*' 'php.*' '^mongodb-.*' '^mysql-.*' \
    azure-cli google-cloud-sdk google-chrome-stable firefox 2>/dev/null || true
sudo apt-get autoremove -y
sudo apt-get clean

# Clean apt cache
echo "Cleaning apt cache..."
sudo rm -rf /var/lib/apt/lists/*

# Remove some large directories
echo "Removing large directories..."
sudo rm -rf /usr/share/dotnet
sudo rm -rf /usr/local/lib/android
sudo rm -rf /opt/ghc
sudo rm -rf /opt/hostedtoolcache/CodeQL

echo "Disk space after cleanup:"
df -h