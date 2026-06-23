param(
    [string]$ObsDir = "E:\OBS",
    [string]$ResultRoot = "E:\clipai_result",
    [string]$DatasetRoot = "E:\Highlights\ml_dataset",
    [int]$Parallel = 3,
    [int]$MaxVideos = 3
)

$ErrorActionPreference = "Continue"
$py = "python"
$infer = "C:\clipAI\files\infer_highlights.py"
$logRoot = Join-Path $ResultRoot "_logs"
New-Item -ItemType Directory -Force -Path $ResultRoot, $logRoot | Out-Null

Write-Host "[batch] waiting for train_binary.py..."
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'train_binary\.py' }
    if (-not $running) { break }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 5

if (-not (Test-Path $infer)) {
    Write-Host "[batch] infer script not found"
    exit 1
}

$files = Get-ChildItem -Path $ObsDir -File -Filter *.mp4 -ErrorAction SilentlyContinue | Sort-Object Name
if (-not $files -or $files.Count -eq 0) {
    Write-Host "[batch] no mp4 files"
    exit 0
}

$allCount = $files.Count
if ($MaxVideos -gt 0) {
    $files = $files | Select-Object -First $MaxVideos
    Write-Host "[batch] pilot: $($files.Count)/$allCount videos (MaxVideos=$MaxVideos) parallel=$Parallel"
} else {
    Write-Host "[batch] full run: $($files.Count) videos parallel=$Parallel"
}

$groupTotal = [Math]::Ceiling($files.Count / $Parallel)

for ($i = 0; $i -lt $files.Count; $i += $Parallel) {
    $end = [Math]::Min($i + $Parallel - 1, $files.Count - 1)
    $batch = $files[$i..$end]
    $procs = @()
    $groupNum = [int]($i / $Parallel) + 1
    Write-Host "[batch] group $groupNum / $groupTotal"

    foreach ($file in $batch) {
        $stem = $file.BaseName
        $outDir = Join-Path $ResultRoot "${stem}_하이라이트"
        $logFile = Join-Path $logRoot ($stem + ".log")
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null

        $argList = @(
            "-u", $infer,
            $file.FullName,
            "--dataset-root", $DatasetRoot,
            "--output-dir", $outDir,
            "--binary-only",
            "--stride-sec", "8",
            "--binary-threshold", "0.55"
        )
        Write-Host "  start: $($file.Name)"
        $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow `
            -RedirectStandardOutput $logFile -RedirectStandardError $logFile
        $procs += $proc
    }

    foreach ($p in $procs) {
        Wait-Process -Id $p.Id -ErrorAction SilentlyContinue
    }
    Write-Host "[batch] group $groupNum done"
}

if ($MaxVideos -gt 0) {
    Write-Host "[batch] pilot complete ($($files.Count) videos) -> $ResultRoot"
    Write-Host "[batch] review results, then run with -MaxVideos 0 for full OBS run"
} else {
    Write-Host "[batch] finished: $ResultRoot"
}
