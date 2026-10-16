@echo off
setlocal EnableExtensions
chcp 65001 >nul
title CoverSong Studio
cd /d "%~dp0"

rem  CoverSong Studio. The first run installs everything into this folder (tools\ and .venv\): uv, Python 3.12,
rem  the libraries, ffmpeg. Later runs only check and start. Nothing is installed system-wide;
rem  deleting the folder removes it all.  Arguments are passed to the app, e.g.  start.bat --studio

set "HERE=%~dp0"
set "TOOLS=%HERE%tools"
set "VENV=%HERE%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "LOGS=%HERE%logs"
set "SETUP_LOG=%HERE%logs\setup.txt"
set "UV=%HERE%tools\uv\uv.exe"
set "UV_CACHE_DIR=%HERE%tools\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%HERE%tools\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
set "UV_HTTP_TIMEOUT=300"
set "PYTHONUTF8=1"
set "NO_PROXY=localhost,127.0.0.1,::1"
set "no_proxy=localhost,127.0.0.1,::1"
set "PATH=%HERE%tools\ffmpeg\bin;%PATH%"
if not exist "%TOOLS%" mkdir "%TOOLS%"
if not exist "%LOGS%" mkdir "%LOGS%"

echo.
echo   CoverSong Studio
echo   ================
echo.

rem ---------------------------------------------------------------- 1. uv: installs Python and the libraries
if exist "%UV%" goto :have_uv
echo [setup 1/4] Downloading uv, the Python package installer (about 20 MB)...
echo === uv %date% %time% === >> "%SETUP_LOG%"
call :download "https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-pc-windows-msvc.zip" "%TOOLS%\uv.zip"
if errorlevel 1 goto :fail
call :unzip "%TOOLS%\uv.zip" "%TOOLS%\uv"
if errorlevel 1 goto :fail
del /q "%TOOLS%\uv.zip" 2>nul
if exist "%UV%" goto :have_uv
for /r "%TOOLS%\uv" %%F in (uv.exe) do if exist "%%F" copy /y "%%F" "%UV%" >nul
if not exist "%UV%" goto :fail
:have_uv

rem ---------------------------------------------------------------- 2. Python 3.12 in .venv
if not exist "%PY%" goto :make_venv
"%PY%" -c "import sys" >nul 2>&1
if not errorlevel 1 goto :have_venv
echo The Python environment does not start (was the folder moved?) - making it again.
rmdir /s /q "%VENV%"
:make_venv
echo [setup 2/4] Python 3.12 (downloaded into tools\python)...
echo === python %date% %time% === >> "%SETUP_LOG%"
"%UV%" venv --python 3.12 "%VENV%" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 goto :fail
:have_venv

rem ---------------------------------------------------------------- 3. the libraries
rem  PyTorch build: CUDA 12.8 for NVIDIA GPUs from the RTX 20 series on; older ones (GTX 10xx and earlier, compute
rem  capability below 7.5) need PyTorch 2.7.1 with CUDA 12.6; no NVIDIA GPU - the CPU build.
set "TORCH=cpu"
set "OVERRIDE="
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :torch_chosen
set "TORCH=cu128"
set "CAP="
for /f "tokens=1 delims=, " %%C in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do if not defined CAP set "CAP=%%C"
if not defined CAP goto :torch_chosen
set "CAPMAJ=" & set "CAPMIN="
for /f "tokens=1,2 delims=." %%M in ("%CAP%") do set "CAPMAJ=%%M" & set "CAPMIN=%%N"
if not defined CAPMIN set "CAPMIN=0"
set "LEGACY="
if %CAPMAJ% LSS 7 set "LEGACY=1"
if %CAPMAJ% EQU 7 if %CAPMIN% LSS 5 set "LEGACY=1"
if not defined LEGACY goto :torch_chosen
set "TORCH=cu126"
(echo torch==2.7.1& echo torchaudio==2.7.1)> "%TOOLS%\torch-legacy.txt"
set "OVERRIDE=--override "%TOOLS%\torch-legacy.txt""
:torch_chosen
set "STAMP=%VENV%\installed-%TORCH%.txt"
if not exist "%STAMP%" goto :install
fc /b "%STAMP%" "%HERE%requirements.txt" >nul 2>&1
if not errorlevel 1 goto :have_libs
:install
set "REINSTALL="
for %%S in ("%VENV%\installed-*.txt") do if /i not "%%~nxS"=="installed-%TORCH%.txt" set "REINSTALL=--reinstall-package torch --reinstall-package torchaudio"
del /q "%VENV%\installed-*.txt" 2>nul
if "%TORCH%"=="cpu" echo [setup 3/4] Libraries. No NVIDIA GPU found: PyTorch for the CPU - everything works, but slowly.
if not "%TORCH%"=="cpu" echo [setup 3/4] Libraries with PyTorch for your NVIDIA GPU, build %TORCH%. The first time this downloads about 5 GB
echo             and takes 5-30 minutes. If it stops, run start.bat again - it continues.
echo === libraries (%TORCH%) %date% %time% === >> "%SETUP_LOG%"
"%UV%" pip install --python "%PY%" -r "%HERE%requirements.txt" --extra-index-url https://download.pytorch.org/whl/%TORCH% --index-strategy unsafe-best-match %OVERRIDE% %REINSTALL%
if not errorlevel 1 goto :check_libs
echo Writing the details to logs\setup.txt...
"%UV%" pip install --python "%PY%" -r "%HERE%requirements.txt" --extra-index-url https://download.pytorch.org/whl/%TORCH% --index-strategy unsafe-best-match %OVERRIDE% %REINSTALL% >> "%SETUP_LOG%" 2>&1
goto :fail
:check_libs
"%PY%" -m coversong.selfcheck --quick > "%LOGS%\selfcheck.txt" 2>&1
set "SC=%errorlevel%"
type "%LOGS%\selfcheck.txt" >> "%SETUP_LOG%"
if "%SC%"=="0" goto :libs_ok
type "%LOGS%\selfcheck.txt"
goto :fail
:libs_ok
copy /y "%HERE%requirements.txt" "%STAMP%" >nul
:have_libs
rem  uv's download cache (3-5 GB) only speeds up a reinstall: the libraries are copied into .venv, so it goes
if exist "%HERE%tools\uv-cache" (
  echo Removing the download cache - the libraries are already installed...
  "%UV%" cache clean >> "%SETUP_LOG%" 2>&1
)

rem ---------------------------------------------------------------- 4. ffmpeg
where ffmpeg >nul 2>&1
if not errorlevel 1 goto :have_ffmpeg
echo [setup 4/4] Downloading ffmpeg (about 150 MB)...
echo === ffmpeg %date% %time% === >> "%SETUP_LOG%"
call :download "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" "%TOOLS%\ffmpeg.zip"
if errorlevel 1 goto :fail
if exist "%TOOLS%\ffmpeg-tmp" rmdir /s /q "%TOOLS%\ffmpeg-tmp"
call :unzip "%TOOLS%\ffmpeg.zip" "%TOOLS%\ffmpeg-tmp"
if errorlevel 1 goto :fail
if exist "%TOOLS%\ffmpeg" rmdir /s /q "%TOOLS%\ffmpeg"
for /d %%D in ("%TOOLS%\ffmpeg-tmp\*") do move "%%D" "%TOOLS%\ffmpeg" >nul
rmdir /s /q "%TOOLS%\ffmpeg-tmp" 2>nul
del /q "%TOOLS%\ffmpeg.zip" 2>nul
if not exist "%TOOLS%\ffmpeg\bin\ffmpeg.exe" goto :fail
:have_ffmpeg

rem ---------------------------------------------------------------- start
echo.
echo Starting... the browser opens by itself. Keep this window open while you work; close it to stop the app.
echo.
"%PY%" -m coversong %*
if not errorlevel 1 exit /b 0
echo.
echo The app stopped with an error - the details are in logs\last_run.txt
pause
exit /b 1

rem ---------------------------------------------------------------- helpers
:download
where curl >nul 2>&1
if errorlevel 1 goto :download_ps
curl -L --fail --retry 3 --retry-delay 3 -o "%~2" "%~1"
if not errorlevel 1 exit /b 0
:download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri '%~1' -OutFile '%~2'" >> "%SETUP_LOG%" 2>&1
exit /b %errorlevel%

:unzip
if not exist "%~2" mkdir "%~2"
tar -xf "%~1" -C "%~2" >> "%SETUP_LOG%" 2>&1
if not errorlevel 1 exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force -LiteralPath '%~1' -DestinationPath '%~2'" >> "%SETUP_LOG%" 2>&1
exit /b %errorlevel%

:fail
echo.
echo [ERROR] Setup did not finish. The details are in logs\setup.txt
echo Check the internet connection and run start.bat again - it continues where it stopped.
pause
exit /b 1
