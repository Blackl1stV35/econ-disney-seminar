@echo off
setlocal
cd /d "%~dp0"
if not exist results mkdir results
set LOG=results\launcher_log.txt
echo ==== run_full.bat started %DATE% %TIME% ==== > "%LOG%"

set PY=
py -3 -c "import sys; print(sys.version)" >> "%LOG%" 2>&1 && set PY=py -3
if "%PY%"=="" (
  python -c "import sys; print(sys.version)" >> "%LOG%" 2>&1 && set PY=python
)
if "%PY%"=="" (
  echo ERROR: Python 3 was not found. >> "%LOG%"
  echo ERROR: Python 3 was not found on this PC. See %LOG%
  pause
  exit /b 1
)
echo Using: %PY% >> "%LOG%"

%PY% -c "import pandas, numpy; print('pandas', pandas.__version__, 'numpy', numpy.__version__)" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo pandas or numpy missing; installing for this user >> "%LOG%"
  echo Installing pandas and numpy for this user...
  %PY% -m pip install --user pandas numpy >> "%LOG%" 2>&1
)

echo Step 1 of 2: checking the SEC file layout...
%PY% run_13f.py inspect 2> results\errors_inspect.txt
if errorlevel 1 (
  echo inspect failed >> "%LOG%"
  echo Step 1 failed. Details are in results\errors_inspect.txt
  pause
  exit /b 1
)
echo Step 2 of 2: downloading and processing the 13F data (about 15-30 minutes)...
%PY% run_13f.py full 2> results\errors_full.txt
echo ==== finished %DATE% %TIME% exit code %ERRORLEVEL% ==== >> "%LOG%"
echo.
echo Done. Results are in the results folder. You can close this window.
pause
