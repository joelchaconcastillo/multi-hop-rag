#!/bin/bash
# Setup script for multi-hop RAG using UV

set -e

echo "================================"
echo "Multi-Hop RAG Setup with UV"
echo "================================"
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✓ UV installed successfully"
else
    echo "✓ UV is already installed"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "✓ Virtual environment created and activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
uv pip install -e ".[dev]"
echo "✓ Dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - HUGGINGFACE_API_KEY"
else
    echo ""
    echo "✓ .env file already exists"
fi

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi
echo "3. Run the example:"
echo "   python examples/basic_usage.py"
echo ""
