@echo off
rem BIST Jarvis PC uygulamasini tarayicida acar.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run jarvis_app.py --browser.gatherUsageStats false
