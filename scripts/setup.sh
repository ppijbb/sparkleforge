#!/bin/bash

# Local Researcher Setup Script
# This script sets up the Local Researcher environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check version
check_version() {
    local cmd="$1"
    local version_flag="$2"
    local min_version="$3"
    
    if command_exists "$cmd"; then
        local version=$($cmd $version_flag 2>&1 | head -n1)
        echo "$version"
    else
        echo "not_installed"
    fi
}

# Function to compare versions
compare_versions() {
    local version1="$1"
    local version2="$2"
    
    # Remove 'v' prefix if present
    version1=${version1#v}
    version2=${version2#v}
    
    # Split version strings
    IFS='.' read -ra v1 <<< "$version1"
    IFS='.' read -ra v2 <<< "$version2"
    
    # Compare each component
    for i in "${!v1[@]}"; do
        if [[ ${v1[$i]} -gt ${v2[$i]} ]]; then
            return 1
        elif [[ ${v1[$i]} -lt ${v2[$i]} ]]; then
            return -1
        fi
    done
    
    return 0
}

# Function to install Node.js dependencies
install_node_dependencies() {
    print_status "Installing Node.js dependencies..."
    
    if [ -f "package.json" ]; then
        npm install
        print_success "Node.js dependencies installed successfully"
    else
        print_warning "package.json not found, skipping Node.js dependencies"
    fi
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed successfully"
    else
        print_warning "requirements.txt not found, skipping Python dependencies"
    fi
}

# Function to setup virtual environment
setup_python_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
}

# Function to install Gemini CLI
install_gemini_cli() {
    print_status "Installing Gemini CLI..."
    
    if command_exists "gemini"; then
        print_warning "Gemini CLI is already installed"
        return
    fi
    
    # Try to install via npm
    if command_exists "npm"; then
        npm install -g @google/gemini-cli
        print_success "Gemini CLI installed via npm"
    else
        print_error "npm not found. Please install Node.js first."
        exit 1
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p outputs
    mkdir -p logs
    mkdir -p data
    mkdir -p configs
    mkdir -p tests
    
    print_success "Directories created successfully"
}

# Function to setup configuration
setup_configuration() {
    print_status "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Local Researcher Environment Variables
# Copy this file to .env and configure your API keys

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_PROJECT_ID=your_google_project_id_here

# News API Configuration
NEWSAPI_KEY=your_newsapi_key_here

# Tavily Configuration
TAVILY_API_KEY=your_tavily_api_key_here

# Perplexity Configuration
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Application Configuration
NODE_ENV=production
LOG_LEVEL=INFO
EOF
        print_success ".env file created"
        print_warning "Please configure your API keys in the .env file"
    else
        print_warning ".env file already exists"
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Node.js
    local node_version=$(check_version "node" "--version" "20.0.0")
    if [ "$node_version" = "not_installed" ]; then
        print_error "Node.js is not installed. Please install Node.js 20.0.0 or higher."
        exit 1
    else
        print_success "Node.js version: $node_version"
    fi
    
    # Check Python
    local python_version=$(check_version "python3" "--version" "3.11.0")
    if [ "$python_version" = "not_installed" ]; then
        print_error "Python 3.11+ is not installed. Please install Python 3.11 or higher."
        exit 1
    else
        print_success "Python version: $python_version"
    fi
    
    # Check npm
    if ! command_exists "npm"; then
        print_error "npm is not installed. Please install npm."
        exit 1
    else
        print_success "npm is installed"
    fi
    
    # Check pip
    if ! command_exists "pip"; then
        print_error "pip is not installed. Please install pip."
        exit 1
    else
        print_success "pip is installed"
    fi
    
    print_success "All prerequisites are satisfied"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python3 -c "
import sys
try:
    import langchain
    import langgraph
    import pydantic
    import rich
    print('Python dependencies: OK')
except ImportError as e:
    print(f'Python dependencies: FAILED - {e}')
    sys.exit(1)
"
    
    # Test Node.js modules
    node -e "
try {
    require('commander');
    require('inquirer');
    require('chalk');
    require('ora');
    console.log('Node.js dependencies: OK');
} catch (e) {
    console.log('Node.js dependencies: FAILED - ' + e.message);
    process.exit(1);
}
"
    
    # Test Gemini CLI
    if command_exists "gemini"; then
        gemini --version > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            print_success "Gemini CLI: OK"
        else
            print_error "Gemini CLI: FAILED"
        fi
    else
        print_error "Gemini CLI: NOT INSTALLED"
    fi
    
    print_success "Installation test completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --check-only        Only check prerequisites"
    echo "  --install-deps      Install dependencies only"
    echo "  --setup-config      Setup configuration only"
    echo "  --test              Test installation only"
    echo ""
    echo "Examples:"
    echo "  $0                  Full setup"
    echo "  $0 --check-only     Check prerequisites only"
    echo "  $0 --install-deps   Install dependencies only"
}

# Main function
main() {
    echo "=========================================="
    echo "    Local Researcher Setup Script"
    echo "=========================================="
    echo ""
    
    # Parse command line arguments
    CHECK_ONLY=false
    INSTALL_DEPS=false
    SETUP_CONFIG=false
    TEST_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --install-deps)
                INSTALL_DEPS=true
                shift
                ;;
            --setup-config)
                SETUP_CONFIG=true
                shift
                ;;
            --test)
                TEST_ONLY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    if [ "$CHECK_ONLY" = true ]; then
        print_success "Prerequisites check completed"
        exit 0
    fi
    
    # Create directories
    create_directories
    
    # Install dependencies
    if [ "$INSTALL_DEPS" = true ] || [ "$CHECK_ONLY" = false ]; then
        install_node_dependencies
        setup_python_venv
        install_python_dependencies
        install_gemini_cli
    fi
    
    # Setup configuration
    if [ "$SETUP_CONFIG" = true ] || [ "$CHECK_ONLY" = false ]; then
        setup_configuration
    fi
    
    # Test installation
    if [ "$TEST_ONLY" = true ] || [ "$CHECK_ONLY" = false ]; then
        test_installation
    fi
    
    echo ""
    echo "=========================================="
    print_success "Local Researcher setup completed!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Configure your API keys in the .env file"
    echo "2. Run: gemini research 'your research topic'"
    echo "3. Check the documentation for more information"
    echo ""
    echo "For help, run: gemini help"
}

# Run main function
main "$@" 