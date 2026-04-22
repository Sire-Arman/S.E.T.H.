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

& $venvPython $scriptPath @extraArgs
