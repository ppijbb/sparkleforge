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

# 프로젝트 루트 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ERA_AGENT_DIR="$PROJECT_ROOT/../open_researcher/ERA/era-agent"

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
    
    # Linux 배포판 확인
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        error "Cannot detect OS. Please install Go manually."
    fi
    
    case $OS in
        ubuntu|debian)
            info "Installing Go via apt..."
            sudo apt update
            sudo apt install -y golang-go
            ;;
        fedora|rhel|centos)
            info "Installing Go via dnf/yum..."
            if command -v dnf &> /dev/null; then
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

# 2. ERA Agent 빌드
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

# 3. ERA Agent 테스트
info "Testing ERA Agent..."
if "$ERA_AGENT_DIR/agent" --help &> /dev/null; then
    success "ERA Agent is working correctly"
else
    error "ERA Agent test failed"
fi

echo ""

# 4. 선택적 의존성 안내 (krunvm, buildah)
info "Checking optional dependencies for full ERA functionality..."

MISSING_DEPS=()

if ! command -v krunvm &> /dev/null; then
    MISSING_DEPS+=("krunvm")
fi

if ! command -v buildah &> /dev/null; then
    MISSING_DEPS+=("buildah")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    info "Optional dependencies not found: ${MISSING_DEPS[*]}"
    info "These are optional but recommended for full ERA functionality."
    echo ""
    read -p "Install optional dependencies? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        case $OS in
            ubuntu|debian)
                if [[ " ${MISSING_DEPS[@]} " =~ " krunvm " ]]; then
                    info "Installing krunvm..."
                    # krunvm 설치 방법은 배포판마다 다를 수 있음
                    sudo apt install -y krunvm || info "krunvm installation failed. Please install manually."
                fi
                if [[ " ${MISSING_DEPS[@]} " =~ " buildah " ]]; then
                    info "Installing buildah..."
                    sudo apt install -y buildah || info "buildah installation failed. Please install manually."
                fi
                ;;
            fedora|rhel|centos)
                if [[ " ${MISSING_DEPS[@]} " =~ " krunvm " ]]; then
                    info "Installing krunvm..."
                    sudo dnf install -y krunvm || sudo yum install -y krunvm || info "krunvm installation failed. Please install manually."
                fi
                if [[ " ${MISSING_DEPS[@]} " =~ " buildah " ]]; then
                    info "Installing buildah..."
                    sudo dnf install -y buildah || sudo yum install -y buildah || info "buildah installation failed. Please install manually."
                fi
                ;;
            *)
                info "Please install ${MISSING_DEPS[*]} manually for your OS"
                ;;
        esac
    fi
else
    success "All optional dependencies are installed"
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

