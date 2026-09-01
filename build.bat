@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/4] 安装依赖...
python -m pip install -q pystray pillow pyinstaller || goto :err

echo [2/4] PyInstaller 打包...
python -m PyInstaller --noconfirm --onedir --windowed --name LLMLauncher --icon icon.ico --add-data "icon.ico;." launcher.py || goto :err

echo [3/4] 复制 llama-server 运行时依赖(本地 bin)...
if not exist "dist\LLMLauncher\bin" mkdir "dist\LLMLauncher\bin"
if exist "bin\*" (
  xcopy /E /Y /Q "bin\*" "dist\LLMLauncher\bin\" >nul
) else (
  echo   [警告] bin 目录为空, llama.cpp 模式将无法启动。
  echo          请从 llama.cpp release 下载 llama-server 及 DLL 放入 bin\
)

echo [4/4] 复制配置文件...
if exist "config.json" (
  copy /Y config.json "dist\LLMLauncher\config.json" >nul
) else (
  copy /Y config.example.json "dist\LLMLauncher\config.json" >nul
  echo   [提示] 未找到 config.json, 已用 config.example.json 作为默认配置
)
if exist "config.d\*.json" (
  if not exist "dist\LLMLauncher\config.d" mkdir "dist\LLMLauncher\config.d"
  copy /Y "config.d\*.json" "dist\LLMLauncher\config.d\" >nul
  echo   [提示] 已复制 config.d\ 下的模型配置
)

echo.
echo 打包完成: dist\LLMLauncher\LLMLauncher.exe
pause
exit /b 0

:err
echo 打包失败!
pause
exit /b 1
