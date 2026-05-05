#!/bin/bash
# SparkleForge Installation Script
# 자동으로 모든 의존성을 설치하고 ERA Agent를 빌드합니다.

set -e  # 에러 발생 시 중단

echo "🚀 SparkleForge Installation Script"
echo "===================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 에러 메시지
error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

# 함수: 성공 메시지
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 함수: 정보 메시지
info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
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

install_with_apt() {
    sudo apt-get update
    sudo apt-get install -y "$@"
}

apt_package_exists() {
    apt-cache show "$1" >/dev/null 2>&1
}

install_era_runtime_dependencies() {
    info "Installing ERA runtime dependencies (krunvm, buildah)..."

    missing=()
    command_exists krunvm || missing+=("krunvm")
    command_exists buildah || missing+=("buildah")

    if [ ${#missing[@]} -eq 0 ]; then
        success "ERA runtime dependencies are already installed"
        return
    fi

    case "$OS" in
        darwin)
            if ! command_exists brew; then
                error "Homebrew is required to install krunvm/buildah on macOS. Install Homebrew first: https://brew.sh/"
            fi
            brew tap slp/krun || true
            brew install "${missing[@]}"
            ;;
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    sudo apt-get update
                    command_exists buildah || sudo apt-get install -y buildah
                    if ! command_exists krunvm; then
                        if apt_package_exists krunvm; then
                            sudo apt-get install -y krunvm
                        else
                            error "krunvm is not available from this Ubuntu/Debian apt repository. Install krunvm from the upstream containers/krunvm instructions, then rerun ./install.sh."
                        fi
                    fi
                    ;;
                fedora)
                    sudo dnf install -y dnf-plugins-core buildah
                    sudo dnf copr enable -y slp/libkrunfw
                    sudo dnf copr enable -y slp/libkrun
                    sudo dnf copr enable -y slp/krunvm
                    sudo dnf install -y krunvm
                    ;;
                rhel|centos)
                    if command_exists dnf; then
                        sudo dnf install -y "${missing[@]}"
                    else
                        sudo yum install -y "${missing[@]}"
                    fi
                    ;;
                arch|manjaro)
                    sudo pacman -S --noconfirm "${missing[@]}"
                    ;;
                *)
                    error "Unsupported Linux distro for automatic krunvm/buildah install: ${DISTRO:-unknown}"
                    ;;
            esac
            ;;
        *)
            error "Unsupported OS for automatic krunvm/buildah install: $OS"
            ;;
    esac

    command_exists krunvm || error "krunvm installation finished but krunvm is still not in PATH"
    command_exists buildah || error "buildah installation finished but buildah is still not in PATH"

    success "ERA runtime dependencies installed"
}

# 프로젝트 루트 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ERA_AGENT_DIR="$PROJECT_ROOT/../open_researcher/ERA/era-agent"
detect_os

# ERA Agent 디렉토리 확인
if [ ! -d "$ERA_AGENT_DIR" ]; then
    error "ERA Agent source not found at $ERA_AGENT_DIR"
fi

info "Project root: $PROJECT_ROOT"
info "ERA Agent dir: $ERA_AGENT_DIR"
echo ""

# 1. Go 설치 확인 및 설치
info "Checking Go installation..."
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | awk '{print $3}')
    success "Go is already installed: $GO_VERSION"
else
    info "Go is not installed. Installing Go..."

    case "$OS" in
        darwin)
            if ! command_exists brew; then
                error "Homebrew is required to install Go on macOS. Install Homebrew first: https://brew.sh/"
            fi
            brew install go
            ;;
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    info "Installing Go via apt..."
                    install_with_apt golang-go
                    ;;
                fedora|rhel|centos)
                    info "Installing Go via dnf/yum..."
                    if command_exists dnf; then
                        sudo dnf install -y golang
                    else
                        sudo yum install -y golang
                    fi
                    ;;
                arch|manjaro)
                    info "Installing Go via pacman..."
                    sudo pacman -S --noconfirm go
                    ;;
                *)
                    error "Unsupported Linux distro: ${DISTRO:-unknown}. Please install Go manually from https://go.dev/dl/"
                    ;;
            esac
            ;;
        *)
            error "Unsupported OS: $OS. Please install Go manually from https://go.dev/dl/"
            ;;
    esac
    
    if command -v go &> /dev/null; then
        GO_VERSION=$(go version | awk '{print $3}')
        success "Go installed successfully: $GO_VERSION"
    else
        error "Go installation failed. Please install manually."
    fi
fi

# Go 버전 확인 (1.21 이상 필요)
GO_MAJOR=$(go version | awk '{print $3}' | sed 's/go//' | cut -d. -f1)
GO_MINOR=$(go version | awk '{print $3}' | sed 's/go//' | cut -d. -f2)
if [ "$GO_MAJOR" -lt 1 ] || ([ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -lt 21 ]); then
    error "Go 1.21 or later is required. Current version: $(go version)"
fi

echo ""

# 2. ERA 런타임 의존성 설치
install_era_runtime_dependencies

echo ""

# 3. ERA Agent 빌드
info "Building ERA Agent..."
cd "$ERA_AGENT_DIR"

if [ ! -f "Makefile" ]; then
    error "Makefile not found in $ERA_AGENT_DIR"
fi

# 기존 바이너리 확인
if [ -f "agent" ] && [ -x "agent" ]; then
    info "ERA Agent binary already exists. Rebuilding..."
fi

# 빌드 실행
if make agent; then
    if [ -f "agent" ] && [ -x "agent" ]; then
        success "ERA Agent built successfully at: $ERA_AGENT_DIR/agent"
        chmod +x agent
    else
        error "Build completed but binary not found"
    fi
else
    error "Failed to build ERA Agent"
fi

echo ""

# 4. ERA Agent 테스트
info "Testing ERA Agent..."
if "$ERA_AGENT_DIR/agent" --help &> /dev/null; then
    success "ERA Agent is working correctly"
else
    error "ERA Agent test failed"
fi

echo ""

echo "===================================="
success "Installation completed successfully!"
echo ""
info "ERA Agent binary location: $ERA_AGENT_DIR/agent"
info "You can now use SparkleForge with ERA code execution."
echo ""
info "To test ERA Agent:"
echo "  $ERA_AGENT_DIR/agent vm temp --language python --cmd \"python -c 'print(\\\"Hello!\\\")'\""
echo ""
info "To start ERA server:"
echo "  $ERA_AGENT_DIR/agent server --addr :8080"
echo ""
info "SparkleForge CLI (설치 후):"
echo "  uv run sparkleforge query \"연구 쿼리\"   # 또는: sparkleforge run \"...\""
echo "  uv run sparkleforge web                 # 웹 대시보드"
echo "  uv run sparkleforge --help              # 전체 도움말"
echo ""
