@echo off
set pwd=%~dp0
echo %pwd% 
rd /q /s %pwd%build
rd /q /s %pwd%install
