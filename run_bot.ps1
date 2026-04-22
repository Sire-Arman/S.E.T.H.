#!/usr/bin/env pwsh
<#
.SYNOPSIS
Runs a Python script using the venv313 virtual environment.
.PARAMETER script
The Python script to run (relative to pipecat_ai folder)
.EXAMPLE
.\run_bot.ps1 server.py
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$script,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$extraArgs
)

$projectDir = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectDir "venv313\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot $script

if (-not (Test-Path $venvPython)) {
    Write-Error "venv313 Python not found at $venvPython"
    exit 1
}

if (-not (Test-Path $scriptPath)) {
    Write-Error "Script not found at $scriptPath"
    exit 1
}

Write-Host "Using Python from venv313..."
Write-Host "Running: $scriptPath"

& $venvPython $scriptPath @extraArgs
