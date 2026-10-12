<#
.SYNOPSIS
    Write a string to a file as UTF-8 WITHOUT BOM.
    Use this instead of Set-Content / Out-File when editing project files.

.EXAMPLE
    $content = Get-FileContent "static\index.html"
    $content = $content -replace "old", "new"
    Write-UTF8 "static\index.html" $content

.EXAMPLE
    # One-liner patch pattern:
    $f = "static\index.html"
    $s = Get-FileContent $f
    $s = $s -replace 'old pattern', 'new value'
    Write-UTF8 $f $s
#>

function Get-FileContent {
    param([string]$Path)
    [System.IO.File]::ReadAllText(
        (Resolve-Path $Path),
        [System.Text.UTF8Encoding]::new($false)   # UTF-8, no BOM
    )
}

function Write-UTF8 {
    param([string]$Path, [string]$Content)
    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        [System.Text.UTF8Encoding]::new($false)   # UTF-8, no BOM  <-- key line
    )
    Write-Host "Written (UTF-8 no BOM): $Path"
}
