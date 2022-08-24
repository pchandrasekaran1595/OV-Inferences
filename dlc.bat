@echo off
setlocal enabledelayedexpansion

title "OpenVINO Model Zoo Dowloader & Converter"

echo.
echo #############################################################################################
echo.
echo Batch Script to download models from OpenVINO Model Zoo anc convert them to IR Representation                
echo.
echo #############################################################################################
echo.

set TARGET=CPU
set TARGET_PRECISION=FP16
set MODEL_FOLDER=OVM
set MODEL_NAME=mobilenet-v3-small-1.0-224-tf

set models_path=%BUILD_DISK%\%MODEL_FOLDER%\models
set models_cache=%BUILD_DISK%\%MODEL_FOLDER%\cache
set irs_path=%BUILD_DISK%\%MODEL_FOLDER%\ir

:input_arguments_loop
if not "%1"=="" (
    if "%1"=="--target" (
        set TARGET=%2
        shift
    )
    if "%1"=="--precision" (
        set TARGET_PRECISION=%2
        shift
    )
    if "%1"=="--model-name" (
        set MODEL_NAME=%2
        shift
    )
    shift
    goto :input_arguments_loop
)

echo Target = %TARGET%
echo.
echo Model Name = %MODEL_NAME%
echo.
echo Precision = %TARGET_PRECISION%
echo.

set ir_dir=%irs_path%\%model_dir%\%target_precision%

cd /d %BUILD_DISK%\Virtual Environments\OpenVINO\Scripts

echo.
echo omz_downloader --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
echo.
omz_downloader --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"

CALL :delay 5
echo.
echo ###############^|^| Run Model Optimizer ^|^|###############
echo.
::set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
echo omz_converter --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
echo.
omz_converter --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
if ERRORLEVEL 1 GOTO errorHandling

goto :eof

:errorHandling
echo Error

:delay
timeout %~1 2> nul
EXIT /B 0