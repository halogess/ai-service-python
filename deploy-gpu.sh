#!/bin/bash

echo "=========================================="
echo "  GPU Deployment Script"
echo "=========================================="

# Check NVIDIA driver
echo ""
echo "1. Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Install NVIDIA driver first!"
    exit 1
fi

nvidia-smi
echo "✅ NVIDIA driver OK"

# Check Docker
echo ""
echo "2. Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install Docker first!"
    exit 1
fi
echo "✅ Docker OK"

# Check NVIDIA Container Toolkit
echo ""
echo "3. Checking NVIDIA Container Toolkit..."
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Container Toolkit not working!"
    echo ""
    echo "Install with:"
    echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    exit 1
fi
echo "✅ NVIDIA Container Toolkit OK"

# Stop existing container
echo ""
echo "4. Stopping existing container..."
docker-compose down

# Build with GPU support
echo ""
echo "5. Building image with GPU support..."
docker-compose build --no-cache

# Start container
echo ""
echo "6. Starting container with GPU..."
docker-compose up -d

# Wait for startup
echo ""
echo "7. Waiting for startup..."
sleep 5

# Check logs
echo ""
echo "8. Checking logs..."
docker logs visual-worker --tail 20

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  - View logs: docker logs visual-worker -f"
echo "  - Check GPU: docker exec visual-worker nvidia-smi"
echo "  - Restart: docker-compose restart"
echo "  - Stop: docker-compose down"
echo ""
