@echo off
setlocal
cd /d "%~dp0"
if not exist results mkdir results
echo Re-running the 13F analysis with the corrected churn measure (about 3-5 minutes; downloads are reused)...
python run_13f_v2.py full 2> results\errors_rerun.txt
echo.
echo Done. Results are in the results folder. You can close this window.
pause
