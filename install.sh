#!/bin/bash
# SparkleForge Installation Script
# Installs the Docker/gVisor sandbox runtime used for safe code execution.

set -e

echo "SparkleForge Installation Script"
echo "================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}$1${NC}"
}

info() {
    echo -e "${YELLOW}$1${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_os() {
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    DISTRO=""

    if [ "$OS" = "linux" ] && [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO="${ID:-}"
    fi
}

install_docker() {
    if command_exists docker; then
        success "Docker is already installed: $(docker --version)"
        return
    fi

    info "Installing Docker..."
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    sudo apt-get update
                    sudo apt-get install -y docker.io
                    ;;
                fedora)
                    sudo dnf install -y docker
                    ;;
                rhel|centos)
                    if command_exists dnf; then
                        sudo dnf install -y docker
                    else
                        sudo yum install -y docker
                    fi
                    ;;
                arch|manjaro)
                    sudo pacman -S --noconfirm docker
                    ;;
                *)
                    error "Unsupported Linux distro for automatic Docker install: ${DISTRO:-unknown}"
                    ;;
            esac
            sudo systemctl enable --now docker || true
            ;;
        darwin)
            error "Install Docker Desktop for macOS, then rerun this script."
            ;;
        *)
            error "Unsupported OS for automatic Docker install: $OS"
            ;;
    esac

    command_exists docker || error "Docker installation finished but docker is still not in PATH"
    success "Docker installed"
}

install_runsc() {
    if command_exists runsc; then
        success "runsc is already installed: $(runsc --version 2>&1 | head -n1)"
    else
        info "Installing gVisor runsc..."
        [ "$OS" = "linux" ] || error "Automatic runsc install is supported only on Linux"

        if ! command_exists curl; then
            case "$DISTRO" in
                ubuntu|debian)
                    sudo apt-get update
                    sudo apt-get install -y ca-certificates curl
                    ;;
                fedora|rhel|centos)
                    if command_exists dnf; then
                        sudo dnf install -y ca-certificates curl
                    else
                        sudo yum install -y ca-certificates curl
                    fi
                    ;;
                arch|manjaro)
                    sudo pacman -S --noconfirm ca-certificates curl
                    ;;
                *)
                    error "curl is required to install runsc"
                    ;;
            esac
        fi

        ARCH="$(uname -m)"
        URL="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}"
        TMPDIR="$(mktemp -d)"
        (
            cd "$TMPDIR"
            curl -fsSLO "${URL}/runsc"
            curl -fsSLO "${URL}/runsc.sha512"
            curl -fsSLO "${URL}/containerd-shim-runsc-v1"
            curl -fsSLO "${URL}/containerd-shim-runsc-v1.sha512"
            sha512sum -c runsc.sha512
            sha512sum -c containerd-shim-runsc-v1.sha512
            chmod a+rx runsc containerd-shim-runsc-v1
            sudo mv runsc containerd-shim-runsc-v1 /usr/local/bin/
        )
        rm -rf "$TMPDIR"
    fi

    command_exists runsc || error "runsc installation finished but runsc is still not in PATH"

    info "Registering runsc as a Docker runtime..."
    sudo runsc install
    if command_exists systemctl; then
        sudo systemctl restart docker
    else
        sudo service docker restart || true
    fi
    success "Docker runtime 'runsc' is registered"
}

verify_sandbox() {
    info "Verifying Docker/gVisor sandbox..."
    docker run --rm --runtime=runsc hello-world >/dev/null
    docker run --rm --runtime=runsc \
        --network none \
        --cpus 1 \
        --memory 512m \
        --pids-limit 128 \
        --read-only \
        --tmpfs /tmp:rw,noexec,nosuid,size=64m \
        python:3.11-slim python -c 'print(1)' >/dev/null
    success "Docker/gVisor sandbox is working"
}

main() {
    detect_os
    install_docker
    install_runsc
    verify_sandbox

    echo ""
    success "Installation completed successfully"
    echo ""
    info "SparkleForge safe code execution now uses Docker with gVisor/runsc."
    echo "  uv sync"
    echo "  uv run sparkleforge --help"
}

main "$@"
