#!/usr/bin/env bash
set -e

REPO="ppijbb/sparkleforge"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BINARY_NAME="sparkleforge"

echo "🔍 Detecting platform..."
OS="$(uname -s)"
ARCH="$(uname -m)"

TARGET=""
if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        TARGET="sparkleforge-linux-amd64"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        TARGET="sparkleforge-linux-arm64"
    fi
elif [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        TARGET="sparkleforge-macos-amd64"
    elif [ "$ARCH" = "arm64" ]; then
        TARGET="sparkleforge-macos-arm64"
    fi
fi

if [ -z "$TARGET" ]; then
    echo "❌ Unsupported platform: $OS / $ARCH"
    exit 1
fi

echo "🚀 Fetching latest SparkleForge release asset ($TARGET)..."
LATEST_URL="https://github.com/$REPO/releases/latest/download/$TARGET"

TMP_FILE="$(mktemp)"
if ! curl -fsSL "$LATEST_URL" -o "$TMP_FILE"; then
    echo "❌ Failed to download SparkleForge binary from $LATEST_URL"
    rm -f "$TMP_FILE"
    exit 1
fi

chmod +x "$TMP_FILE"

echo "📦 Installing sparkleforge to $INSTALL_DIR/$BINARY_NAME..."
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
else
    echo "🔑 Root permissions required to install to $INSTALL_DIR. Asking for sudo..."
    sudo mv "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
fi

echo "🐍 Checking python runtime auto-detect / bootstrap dependencies..."
if ! command -v python3 &>/dev/null; then
    echo "⚠️ Warning: python3 is not detected in PATH. SparkleForge requires Python 3.10+ for execution backends."
fi

echo "✨ SparkleForge has been successfully installed!"
sparkleforge --help || true
