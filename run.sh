#!/bin/bash
# ═══════════════════════════════════════════════════════════
#   NeuroScan Production Launcher - macOS/Linux
#   One-click startup with comprehensive error handling
# ═══════════════════════════════════════════════════════════

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions for formatted output
print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         NEUROSCAN - ALZHEIMER'S MRI CLASSIFICATION         ║"
    echo "║              Production Launcher v2.0                       ║"
    echo "║           EfficientNet-B2 • 99%+ Accuracy                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Error handler
error_exit() {
    print_error "$1"
    echo ""
    print_warning "For support, please check the README.md file"
    exit 1
}

# Change to script directory
cd "$(dirname "$0")" || error_exit "Cannot change to script directory"

print_banner

# Step 1: Check Python installation
print_step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3 is not installed. Please install Python 3.9 or higher."
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Step 2: Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_step "Creating virtual environment..."
    python3 -m venv venv || error_exit "Failed to create virtual environment"
    print_success "Virtual environment created"
fi

# Step 3: Activate virtual environment
print_step "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate || error_exit "Failed to activate virtual environment"
    print_success "Virtual environment activated"
else
    error_exit "Virtual environment not found. Please delete 'venv' folder and try again."
fi

# Step 4: Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded"

# Step 5: Install dependencies
print_step "Checking dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt || error_exit "Failed to install dependencies"
    print_success "Dependencies installed"
else
    error_exit "requirements.txt not found!"
fi

# Step 6: Check if model exists
print_step "Checking model file..."
if [ ! -f "model/best_model_b2.pth" ]; then
    error_exit "Model file not found at model/best_model_b2.pth"
fi
MODEL_SIZE=$(du -h "model/best_model_b2.pth" | cut -f1)
print_success "Model found (size: $MODEL_SIZE)"

# Step 7: Check if port 8000 is available
print_step "Checking port 8000 availability..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port 8000 is already in use"
    read -p "Do you want to kill the existing process and continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Killing existing process..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
        print_success "Process killed"
    else
        error_exit "Port 8000 is occupied. Please free it and try again."
    fi
fi

# Step 8: Detect device
print_step "Detecting acceleration device..."
DEVICE_INFO="CPU"
if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE_INFO="MPS (Apple Silicon GPU)"
    print_success "MPS acceleration detected - using Apple Silicon GPU"
elif python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE_INFO="CUDA (NVIDIA GPU)"
    print_success "CUDA acceleration detected - using NVIDIA GPU"
else
    print_warning "No GPU detected - using CPU (slower performance)"
fi

# Step 9: Start server
echo ""
print_step "Starting NeuroScan server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Server will start at: http://localhost:8000"
echo "  Device: $DEVICE_INFO"
echo "  Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to open browser
open_browser() {
    sleep 3
    print_step "Opening browser..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open http://localhost:8000
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8000
        elif command -v gnome-open &> /dev/null; then
            gnome-open http://localhost:8000
        else
            print_warning "Could not open browser automatically"
            print_warning "Please open http://localhost:8000 in your browser"
        fi
    fi
}

# Start browser in background
open_browser &

# Start server
trap 'echo ""; print_step "Shutting down server..."; exit 0' INT TERM

python3 simple_api.py || error_exit "Server failed to start. Check error messages above."