#!/bin/bash
# Script to free up space on GitHub Actions runner

set -e

echo "Freeing up disk space on GitHub Actions runner..."
df -h

# Remove unnecessary large packages
echo "Removing unnecessary large packages..."
sudo apt-get remove -y '^dotnet-.*'
sudo apt-get remove -y '^llvm-.*'
sudo apt-get remove -y 'php.*'
sudo apt-get remove -y '^mongodb-.*'
sudo apt-get remove -y '^mysql-.*'
sudo apt-get remove -y azure-cli google-cloud-sdk google-chrome-stable firefox
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