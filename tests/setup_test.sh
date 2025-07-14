#!/bin/bash

# Test setup script for morphometry tool
# This script sets up the testing environment and installs dependencies

echo "🔧 Setting up morphometry test environment..."

# Check if we're in the correct directory
if [ ! -f "../src/main.py" ]; then
    echo "❌ Error: Please run this script from the tests directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "test_env" ]; then
    echo "📦 Creating test virtual environment..."
    python3 -m venv test_env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source test_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install main requirements
if [ -f "../requirements.txt" ]; then
    echo "📋 Installing main requirements..."
    pip install -r ../requirements.txt
else
    echo "⚠️ Main requirements.txt not found, installing basic dependencies..."
    pip install numpy opencv-python scikit-image pandas scikit-learn matplotlib seaborn pyyaml
fi

# Install test requirements
if [ -f "requirements-test.txt" ]; then
    echo "🧪 Installing test requirements..."
    pip install -r requirements-test.txt
else
    echo "📋 Installing basic test dependencies..."
    pip install pytest pytest-cov pytest-mock memory-profiler psutil
fi

# Install additional dependencies that might be missing
echo "➕ Installing additional dependencies..."
pip install SimpleITK pydicom joblib tensorflow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Create test data directories
echo "📁 Creating test directories..."
mkdir -p test_data/raw
mkdir -p test_data/processed
mkdir -p test_data/output

# Make test scripts executable
echo "🔐 Making test scripts executable..."
chmod +x quick_test.py
chmod +x test_morphometry.py

# Run quick test to verify setup
echo "🚀 Running quick test..."
python quick_test.py

if [ $? -eq 0 ]; then
    echo "✅ Test environment setup completed successfully!"
    echo ""
    echo "📋 To run tests:"
    echo "   source tests/test_env/bin/activate"
    echo "   python tests/quick_test.py                    # Quick functionality test"
    echo "   python tests/test_morphometry.py --quick      # Quick unit tests"
    echo "   python tests/test_morphometry.py              # Full test suite"
    echo "   python tests/test_morphometry.py --integration # Integration tests only"
else
    echo "❌ Test environment setup failed. Please check the error messages above."
    exit 1
fi