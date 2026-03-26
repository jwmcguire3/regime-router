[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("menu", "plan", "run", "smoke", "models", "list", "list-runs", "show", "show-run", "last")]
    [string]$Command = "menu",

    [Parameter(Position = 1)]
    [string]$Task = "",

    [string]$Model = "dolphin29:latest",
    [string]$Risks = "",
    [string]$BaseUrl = "http://localhost:11434",
    [string]$OutDir = "runs",
    [string]$SaveAs = "",
    [string]$Filename = "",
    [string]$TaskAnalyzerModel = "dolphin29:latest",
    [switch]$NoHandoff,
    [switch]$UseTaskAnalyzer,
    [switch]$DebugRouting,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyFile = Join-Path $ScriptRoot "cognitive_router_prototype.py"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkGray
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkGray
}

function Write-Info {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Gray
}

function Write-Ok {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Green
}

function Write-WarnText {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

function Get-PythonCommand {
    foreach ($candidate in @("py", "python")) {
        try {
            $null = Get-Command $candidate -ErrorAction Stop
            return $candidate
        } catch {
        }
    }
    throw "Could not find Python launcher. Install Python or ensure 'py' or 'python' is on PATH."
}

function Assert-RouterFile {
    if (-not (Test-Path $PyFile)) {
        throw "Could not find cognitive_router_prototype.py at: $PyFile"
    }
}

function Invoke-RouterPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgList,
        [switch]$Capture
    )

    Assert-RouterFile
    $py = Get-PythonCommand
    $fullArgs = @($PyFile) + @($ArgList)

    if ($Capture) {
        $output = & $py @fullArgs 2>&1
        $exitCode = $LASTEXITCODE
        return [pscustomobject]@{
            ExitCode = $exitCode
            Output   = ($output -join [Environment]::NewLine)
        }
    }

    & $py @fullArgs
    return $LASTEXITCODE
}

function Get-CommonArgs {
    return @("--base-url", $BaseUrl, "--out-dir", $OutDir)
}

function Add-PlanFlags {
    param([string[]]$ArgList)

    $result = @($ArgList)

    if ($Risks) {
        $result += @("--risks", $Risks)
    }
    if ($NoHandoff) {
        $result += "--no-handoff"
    }
    if ($UseTaskAnalyzer) {
        $result += "--use-task-analyzer"
        if ($TaskAnalyzerModel) {
            $result += @("--task-analyzer-model", $TaskAnalyzerModel)
        }
    }
    if ($DebugRouting) {
        $result += "--debug-routing"
    }

    return $result
}

function Add-RunFlags {
    param([string[]]$ArgList)

    $result = @($ArgList)

    if ($Risks) {
        $result += @("--risks", $Risks)
    }
    if ($SaveAs) {
        $result += @("--save-as", $SaveAs)
    }
    if ($NoHandoff) {
        $result += "--no-handoff"
    }
    if ($UseTaskAnalyzer) {
        $result += "--use-task-analyzer"
        if ($TaskAnalyzerModel) {
            $result += @("--task-analyzer-model", $TaskAnalyzerModel)
        }
    }

    return $result
}

function Invoke-Plan {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskText
    )

    $argList = Get-CommonArgs
    $argList += @("plan", "--task", $TaskText)
    $argList = Add-PlanFlags -ArgList $argList

    if ($Json) {
        Invoke-RouterPython -ArgList $argList -Capture
    } else {
        Invoke-RouterPython -ArgList $argList
    }
}

function Invoke-Run {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskText
    )

    $argList = Get-CommonArgs
    $argList += @("run", "--task", $TaskText, "--model", $Model)
    $argList = Add-RunFlags -ArgList $argList

    if ($Json) {
        Invoke-RouterPython -ArgList $argList -Capture
    } else {
        Invoke-RouterPython -ArgList $argList
    }
}

function Invoke-Models {
    $argList = (Get-CommonArgs) + @("models")
    Invoke-RouterPython -ArgList $argList
}

function Invoke-ListRuns {
    $argList = (Get-CommonArgs) + @("list-runs")
    Invoke-RouterPython -ArgList $argList
}

function Get-LatestRunFile {
    if (-not (Test-Path $OutDir)) {
        return $null
    }

    $files = Get-ChildItem -Path $OutDir -Filter *.json -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending

    return $files | Select-Object -First 1
}

function Invoke-ShowRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunFileName
    )

    $argList = (Get-CommonArgs) + @("show-run", $RunFileName)
    Invoke-RouterPython -ArgList $argList
}

function Read-BooleanChoice {
    param(
        [string]$Prompt,
        [bool]$Default = $false
    )

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $answer = (Read-Host "$Prompt $suffix").Trim().ToLowerInvariant()

    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $Default
    }

    return $answer -in @("y", "yes", "true", "1")
}

function Invoke-Smoke {
    Write-Section "Smoke tests"

    $smokeCases = @(
        @{
            Name = "Interpretation"
            Task = "What is the strongest interpretation of this pattern?"
            Risks = ""
            UseAnalyzer = $false
            Debug = $false
            Mode = "plan"
        },
        @{
            Name = "Unknowns"
            Task = "What remains unknown or unclear here?"
            Risks = ""
            UseAnalyzer = $false
            Debug = $false
            Mode = "plan"
        },
        @{
            Name = "Stress test"
            Task = "Stress test this frame and identify what would break it."
            Risks = ""
            UseAnalyzer = $false
            Debug = $false
            Mode = "plan"
        },
        @{
            Name = "Decision"
            Task = "We need to decide now between two close options."
            Risks = "optionality"
            UseAnalyzer = $false
            Debug = $false
            Mode = "plan"
        },
        @{
            Name = "Reusable pattern"
            Task = "This looks repeatable and should become a reusable system."
            Risks = ""
            UseAnalyzer = $false
            Debug = $false
            Mode = "plan"
        },
        @{
            Name = "Low-confidence routing with analyzer"
            Task = "There are many signals, no center, some uncertainty, and we may need to choose soon."
            Risks = ""
            UseAnalyzer = $true
            Debug = $true
            Mode = "plan"
        }
    )

    $passCount = 0
    $failCount = 0

    foreach ($case in $smokeCases) {
        Write-Host ""
        Write-Host ("[{0}] {1}" -f $case.Name, $case.Task) -ForegroundColor White

        $caseArgs = Get-CommonArgs
        $caseArgs += @($case.Mode, "--task", $case.Task)

        if ($case.Risks) {
            $caseArgs += @("--risks", $case.Risks)
        }
        if ($NoHandoff) {
            $caseArgs += "--no-handoff"
        }
        if ($case.UseAnalyzer) {
            $caseArgs += "--use-task-analyzer"
            if ($TaskAnalyzerModel) {
                $caseArgs += @("--task-analyzer-model", $TaskAnalyzerModel)
            }
        }
        if ($case.Debug) {
            $caseArgs += "--debug-routing"
        }

        $result = Invoke-RouterPython -ArgList $caseArgs -Capture

        if ($result.ExitCode -eq 0) {
            Write-Ok "PASS"
            $passCount++
            $preview = ($result.Output -split "`r?`n" | Select-Object -First 12) -join [Environment]::NewLine
            Write-Host $preview -ForegroundColor DarkGray
        } else {
            Write-Fail "FAIL"
            $failCount++
            Write-Host $result.Output -ForegroundColor Red
        }
    }

    Write-Section "Smoke summary"
    Write-Host ("Passed: {0}" -f $passCount) -ForegroundColor Green
    Write-Host ("Failed: {0}" -f $failCount) -ForegroundColor Red
}

function Show-Menu {
    while ($true) {
        Write-Section "Router menu"
        Write-Host "1. Plan task"
        Write-Host "2. Run task against Ollama"
        Write-Host "3. List models"
        Write-Host "4. List saved runs"
        Write-Host "5. Show latest run"
        Write-Host "6. Show named run"
        Write-Host "7. Smoke tests"
        Write-Host "8. Exit"
        Write-Host ""

        $choice = Read-Host "Choose an option"

        switch ($choice) {
            "1" {
                $taskText = Read-Host "Task"
                $riskText = Read-Host "Risks (comma-separated, optional)"
                $useAnalyzerChoice = Read-BooleanChoice -Prompt "Use task analyzer" -Default $UseTaskAnalyzer.IsPresent
                $debugChoice = Read-BooleanChoice -Prompt "Debug routing output" -Default $DebugRouting.IsPresent
                $analyzerModelText = if ($useAnalyzerChoice) { Read-Host "Task analyzer model [$TaskAnalyzerModel]" } else { "" }
                if ([string]::IsNullOrWhiteSpace($analyzerModelText)) {
                    $analyzerModelText = $TaskAnalyzerModel
                }
                if ($useAnalyzerChoice) { $script:TaskAnalyzerModel = $analyzerModelText }
                $script:Risks = $riskText
                $script:UseTaskAnalyzer = [bool]$useAnalyzerChoice
                $script:DebugRouting = [bool]$debugChoice
                Invoke-Plan -TaskText $taskText
                Pause
            }
            "2" {
                $taskText = Read-Host "Task"
                $modelText = Read-Host "Model [$Model]"
                if ($modelText) { $script:Model = $modelText }
                $riskText = Read-Host "Risks (comma-separated, optional)"
                $saveText = Read-Host "Save as (optional)"
                $useAnalyzerChoice = Read-BooleanChoice -Prompt "Use task analyzer" -Default $UseTaskAnalyzer.IsPresent
                $analyzerModelText = if ($useAnalyzerChoice) { Read-Host "Task analyzer model [$TaskAnalyzerModel]" } else { "" }
                if ([string]::IsNullOrWhiteSpace($analyzerModelText)) {
                    $analyzerModelText = $TaskAnalyzerModel
                }
                if ($useAnalyzerChoice) { $script:TaskAnalyzerModel = $analyzerModelText }
                $script:Risks = $riskText
                $script:SaveAs = $saveText
                $script:UseTaskAnalyzer = [bool]$useAnalyzerChoice
                Invoke-Run -TaskText $taskText
                Pause
            }
            "3" {
                Invoke-Models
                Pause
            }
            "4" {
                Invoke-ListRuns
                Pause
            }
            "5" {
                $latest = Get-LatestRunFile
                if ($null -eq $latest) {
                    Write-WarnText "No saved runs found."
                } else {
                    Write-Info ("Showing latest run: {0}" -f $latest.Name)
                    Invoke-ShowRun -RunFileName $latest.Name
                }
                Pause
            }
            "6" {
                $runName = Read-Host "Saved run filename"
                if (-not [string]::IsNullOrWhiteSpace($runName)) {
                    Invoke-ShowRun -RunFileName $runName
                } else {
                    Write-WarnText "No filename provided."
                }
                Pause
            }
            "7" {
                Invoke-Smoke
                Pause
            }
            "8" {
                return
            }
            default {
                Write-WarnText "Invalid choice."
                Pause
            }
        }
    }
}

try {
    switch ($Command) {
        "menu" {
            Show-Menu
        }
        "plan" {
            if (-not $Task) { throw 'Provide a task, e.g. .\router.ps1 plan "What is the strongest interpretation?"' }
            Invoke-Plan -TaskText $Task
        }
        "run" {
            if (-not $Task) { throw 'Provide a task, e.g. .\router.ps1 run "What is the strongest interpretation?"' }
            Invoke-Run -TaskText $Task
        }
        "models" {
            Invoke-Models
        }
        "list" {
            Invoke-ListRuns
        }
        "list-runs" {
            Invoke-ListRuns
        }
        "show" {
            if (-not $Filename) { throw "Use -Filename with the saved run filename." }
            Invoke-ShowRun -RunFileName $Filename
        }
        "show-run" {
            if (-not $Filename) { throw "Use -Filename with the saved run filename." }
            Invoke-ShowRun -RunFileName $Filename
        }
        "last" {
            $latest = Get-LatestRunFile
            if ($null -eq $latest) {
                throw "No saved runs found in '$OutDir'."
            }
            Invoke-ShowRun -RunFileName $latest.Name
        }
        "smoke" {
            Invoke-Smoke
        }
        default {
            throw "Unknown command: $Command"
        }
    }
}
catch {
    Write-Fail $_.Exception.Message
    exit 1
}
