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
set MODEL_FOLDER=OVM

set models_path=%BUILD_DISK%\%MODEL_FOLDER%\models
set models_cache=%BUILD_DISK%\%MODEL_FOLDER%\cache
set irs_path=%BUILD_DISK%\%MODEL_FOLDER%\ir

:input_arguments_loop
if not "%1"=="" (
    if "%1"=="--target" (
        set TARGET=%2
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

set ir_dir=%irs_path%\%model_dir%\%target_precision%

cd /d %BUILD_DISK%\Virtual Environments\OpenVINO\Scripts

echo.
echo omz_downloader --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
echo.
omz_downloader --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"

goto :eof

:errorHandling
echo Error

:delay
timeout %~1 2> nul
EXIT /B 0