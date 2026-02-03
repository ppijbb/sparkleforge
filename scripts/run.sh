#!/bin/bash

# Local Researcher - Run Script
# This script sets up and runs the Local Researcher system

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

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python version: $python_version"
        
        # Check if version is 3.11 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
            return 0
        else
            print_error "Python 3.11 or higher is required. Current: $python_version"
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        return 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    print_success "Python dependencies installed"
    
    print_status "Installing Node.js dependencies..."
    if command_exists npm; then
        npm install
        print_success "Node.js dependencies installed"
    else
        print_warning "npm not found, skipping Node.js dependencies"
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p outputs
    mkdir -p logs
    mkdir -p data
    mkdir -p templates
    print_success "Directories created"
}

# Function to setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Local Researcher Environment Variables (v2.0 - 8대 혁신)
# Copy this file to .env and configure your API keys

# OpenRouter Configuration (필수)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# LLM Configuration (Gemini 2.5 Flash Lite 우선)
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-2.5-flash-lite
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000

# Multi-Model Orchestration (Gemini 2.5 Flash Lite)
PLANNING_MODEL=google/gemini-2.5-flash-lite
REASONING_MODEL=google/gemini-2.5-flash-lite
VERIFICATION_MODEL=google/gemini-2.5-flash-lite
GENERATION_MODEL=google/gemini-2.5-flash-lite
COMPRESSION_MODEL=google/gemini-2.5-flash-lite

# MCP Configuration (MCP Only - No Fallbacks)
MCP_ENABLED=true
MCP_TIMEOUT=30
ENABLE_AUTO_FALLBACK=false

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

# Function to run the application
run_app() {
    print_status "Starting Local Researcher..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run the application
    python main.py "$@"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND] [ARGS...]"
    echo ""
    echo "Options:"
    echo "  --setup-only    Only setup the environment, don't run"
    echo "  --help          Show this help message"
    echo ""
    echo "Commands:"
    echo "  research <topic> [options]  - Start a new research project"
    echo "  status [research_id]        - Check research status"
    echo "  list [--status=STATUS]      - List research projects"
    echo "  cancel <research_id>        - Cancel a research project"
    echo "  interactive                 - Run in interactive mode"
    echo "  help                        - Show help"
    echo ""
    echo "Examples:"
    echo "  $0 --setup-only"
    echo "  $0 research 'Artificial Intelligence trends'"
    echo "  $0 research 'Climate change' --domain=science --depth=comprehensive"
    echo "  $0 status research_20240101_1234_5678"
    echo "  $0 list --status=completed"
    echo "  $0 interactive"
}

# Main function
main() {
    echo "=========================================="
    echo "    Local Researcher - Run Script"
    echo "=========================================="
    echo ""
    
    # Parse command line arguments
    SETUP_ONLY=false
    HELP=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                SETUP_ONLY=true
                shift
                ;;
            --help)
                HELP=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [ "$HELP" = true ]; then
        show_usage
        exit 0
    fi
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Setup virtual environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Setup configuration
    setup_config
    
    if [ "$SETUP_ONLY" = true ]; then
        print_success "Setup completed successfully!"
        echo ""
        echo "Next steps:"
        echo "1. Configure your API keys in the .env file"
        echo "2. Run: $0 research 'your research topic'"
        echo "3. Or run: $0 interactive"
        exit 0
    fi
    
    # Run the application
    run_app "$@"
}

# Run main function
main "$@"
