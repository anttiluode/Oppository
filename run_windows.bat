@echo off
python3.13 -c "import torch; print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA BUILD')"
if errorlevel 1 pause & exit /b 1
python3.13 app.py
if errorlevel 1 pause
