@echo off
REM ============================================================
REM  Elimina los modelos serializados (.pkl) de la carpeta modelos\.
REM  Utiles para liberar espacio o antes de comprimir/entregar.
REM  Se regeneran al ejecutar la Fase 6 del notebook.
REM ============================================================
cd /d "%~dp0"
if exist "modelos\*.pkl" (
    del /q "modelos\*.pkl"
    echo Modelos .pkl eliminados de la carpeta modelos\.
) else (
    echo No habia modelos .pkl que eliminar.
)
pause
