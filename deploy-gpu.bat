@echo off
echo ==========================================
echo   GPU Deployment Script (Windows)
echo ==========================================

echo.
echo 1. Checking NVIDIA driver...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo X nvidia-smi not found. Install NVIDIA driver first!
    pause
    exit /b 1
)
nvidia-smi
echo √ NVIDIA driver OK

echo.
echo 2. Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo X Docker not found. Install Docker Desktop first!
    pause
    exit /b 1
)
echo √ Docker OK

echo.
echo 3. Stopping existing container...
docker-compose down

echo.
echo 4. Building image with GPU support...
docker-compose build --no-cache

echo.
echo 5. Starting container with GPU...
docker-compose up -d

echo.
echo 6. Waiting for startup...
timeout /t 5 /nobreak >nul

echo.
echo 7. Checking logs...
docker logs visual-worker --tail 20

echo.
echo ==========================================
echo   Deployment Complete!
echo ==========================================
echo.
echo Commands:
echo   - View logs: docker logs visual-worker -f
echo   - Check GPU: docker exec visual-worker nvidia-smi
echo   - Restart: docker-compose restart
echo   - Stop: docker-compose down
echo.
pause
