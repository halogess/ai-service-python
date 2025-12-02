# Setup GPU untuk Docker

## Prerequisites
1. NVIDIA GPU (GTX/RTX series atau Tesla)
2. NVIDIA Driver terinstall di host
3. Docker versi 19.03+

## Install NVIDIA Container Toolkit

### Ubuntu/Debian:
```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Windows (WSL2):
```bash
# Install NVIDIA Driver for WSL2 dari:
# https://developer.nvidia.com/cuda/wsl

# Install NVIDIA Container Toolkit di WSL2:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Verify GPU Access

```bash
# Test NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Deploy dengan GPU

```bash
# Rebuild image
docker-compose build

# Run dengan GPU
docker-compose up -d

# Check logs
docker logs visual-worker -f

# Verify GPU usage
docker exec visual-worker nvidia-smi
```

## Expected Output
```
Loading LayoutLMv3 model...
Model loaded on cuda
```

## Troubleshooting

### Error: "could not select device driver"
```bash
sudo systemctl restart docker
```

### Error: "nvidia-smi not found in container"
- Pastikan base image menggunakan `nvidia/cuda:*-runtime-*`

### GPU tidak terdeteksi
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker runtime
docker info | grep -i runtime
```
