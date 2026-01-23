@echo off
chcp 65001 > nul

set PYTHONUTF8=1

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] 正在激活虚拟环境 .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] 未找到 .venv 文件夹，将尝试直接执行。
)

if not exist "logs" mkdir logs

:: 使用 PowerShell 获取时间戳 (格式: 20231027_153000)
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TIMESTAMP=%%i
set LOG_FILE=logs\log_%TIMESTAMP%.txt

echo [INFO] 程序已启动，日志将同步显示并记录到 %LOG_FILE%...

powershell -Command "$OutputEncoding = [System.Text.Encoding]::UTF8; $ErrorActionPreference = 'Continue'; python -u main.py 2>&1 | Tee-Object -FilePath %LOG_FILE%"

echo [INFO] 程序运行结束。
pause
