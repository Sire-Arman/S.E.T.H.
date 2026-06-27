#!/usr/bin/env pwsh
<#
.SYNOPSIS
Runs a Python script using the project's .venv virtual environment.
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

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot $script

if (-not (Test-Path $venvPython)) {
    Write-Error ".venv not found. Create it with: py -3.13 -m venv .venv"
    exit 1
}

if (-not (Test-Path $scriptPath)) {
    Write-Error "Script not found at $scriptPath"
    exit 1
}

Write-Host "Using Python from .venv..."
Write-Host "Running: $scriptPath"

# Force UTF-8 everywhere — prevents 'charmap' codec errors on Windows
# when the LLM returns emoji / special Unicode characters.
$env:PYTHONUTF8        = "1"          # Python 3.7+ UTF-8 mode (PEP 540)
$env:PYTHONIOENCODING  = "utf-8"      # Fallback for older sub-processes
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # PowerShell terminal

& $venvPython $scriptPath @extraArgs
