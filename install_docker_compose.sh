#!/bin/bash

# This script installs Docker Compose cleanly.
# It determines the latest version and your system's architecture
# to download the appropriate binary.

set -e

# Check for root privileges
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Please use sudo." >&2
  exit 1
fi

# Determine the latest Docker Compose version
LATEST_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')

if [ -z "$LATEST_VERSION" ]; then
    echo "Failed to fetch the latest Docker Compose version. Please check your internet connection and try again." >&2
    exit 1
fi

# Determine OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Adjust architecture naming for compatibility
case $ARCH in
    x86_64) ARCH="x86_64" ;;
    aarch64) ARCH="aarch64" ;;
    armv7l) ARCH="armv7" ;;
    *)
      echo "Unsupported architecture: $ARCH" >&2
      exit 1
      ;;
esac

# Set the download URL and installation path
DOWNLOAD_URL="https://github.com/docker/compose/releases/download/${LATEST_VERSION}/docker-compose-${OS}-${ARCH}"
INSTALL_PATH="/usr/local/bin/docker-compose"

echo "Downloading Docker Compose version ${LATEST_VERSION} for ${OS}-${ARCH}..."

# Download the binary
if ! curl -L -o "${INSTALL_PATH}" "${DOWNLOAD_URL}"; then
    echo "Failed to download Docker Compose. Please check the URL and your network connection." >&2
    exit 1
fi

# Make the binary executable
chmod +x "${INSTALL_PATH}"

# Verify the installation
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose could not be found after installation. Please check your system's PATH." >&2
    exit 1
fi

INSTALLED_VERSION=$(docker-compose version --short)

echo ""
echo "Docker Compose version ${INSTALLED_VERSION} has been installed successfully to ${INSTALL_PATH}"