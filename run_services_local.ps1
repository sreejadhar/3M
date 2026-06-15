# Launches all DataNanite services locally from the .venv314 virtualenv.
# Each service runs in its own background process; stdout/stderr -> logs\<svc>.log
# Usage:  .\run_services_local.ps1            (start all)
#         .\run_services_local.ps1 -Stop      (stop all)

param([switch]$Stop)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root ".venv314\Scripts\python.exe"
$logs = Join-Path $root "logs"
$pidFile = Join-Path $logs "service_pids.txt"

$servicePorts = 8000,8001,8002,8003,8004,8005,8006,8007,8008,8501

if ($Stop) {
  # The venv python.exe is a launcher shim, so recorded PIDs aren't the uvicorn
  # workers. Kill whatever is actually listening on the service ports instead.
  $owners = @()
  foreach ($p in $servicePorts) {
    Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue |
      ForEach-Object { $owners += $_.OwningProcess }
  }
  $owners = $owners | Sort-Object -Unique
  if ($owners) {
    foreach ($procId in $owners) {
      $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
      if ($proc) { "stopped PID $procId ($($proc.ProcessName))"; Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }
    }
  } else { "No services listening on the target ports." }
  if (Test-Path $pidFile) { Remove-Item $pidFile -Force }
  return
}

New-Item -ItemType Directory -Force -Path $logs | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "data")    | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "reports") | Out-Null

# --- Load .env into this process so child processes inherit it ---
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
      $idx = $line.IndexOf("=")
      $k = $line.Substring(0, $idx).Trim()
      $v = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
      [Environment]::SetEnvironmentVariable($k, $v, "Process")
    }
  }
}

# --- Local wiring: services talk to each other over localhost ---
$env:ORCHESTRATOR_URL = "http://localhost:8005"
$env:METADATA_API_URL = "http://localhost:8000"
$env:ONTOLOGY_API_URL = "http://localhost:8001"
$env:KG_API_URL       = "http://localhost:8002"
$env:DIALOG_API_URL   = "http://localhost:8003"
$env:CONFORMITY_API_URL = "http://localhost:8004"
$env:SHACL_API_URL      = "http://localhost:8007"
$env:UNSTRUCTURED_API_URL = "http://localhost:8008"
$env:AGENT_API_URL    = "http://localhost:8000"
$env:DATA_DIR         = (Join-Path $root "data")

# service-name -> uvicorn target & port
$services = @(
  @{ name = "agent-api";        app = "api:app";              port = 8000 },
  @{ name = "ontology-api";     app = "ontology_api:app";     port = 8001 },
  @{ name = "kg-api";           app = "kg_api:app";           port = 8002 },
  @{ name = "dialog-api";       app = "dialog_api:app";       port = 8003 },
  @{ name = "conformity-api";   app = "conformity_api:app";   port = 8004 },
  @{ name = "chat-ui";          app = "orchestrator_api:app"; port = 8005 },
  @{ name = "tech-ui";          app = "tech_ui_server:app";   port = 8006 },
  @{ name = "shacl-api";        app = "shacl_api:app";        port = 8007 },
  @{ name = "unstructured-api"; app = "unstructured_api:app"; port = 8008 }
)

if (Test-Path $pidFile) { Remove-Item $pidFile -Force }

foreach ($s in $services) {
  $log = Join-Path $logs "$($s.name).log"
  $args = @("-m", "uvicorn", $s.app, "--host", "127.0.0.1", "--port", "$($s.port)", "--log-level", "info")
  $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -WindowStyle Hidden -PassThru
  $p.Id | Out-File -FilePath $pidFile -Append -Encoding ascii
  "started {0,-18} port {1}  PID {2}" -f $s.name, $s.port, $p.Id
}

# --- Streamlit UI (port 8501) ---
$slog = Join-Path $logs "streamlit-ui.log"
$slArgs = @("-m", "streamlit", "run", "app.py", "--server.address=127.0.0.1",
            "--server.port=8501", "--server.headless=true",
            "--server.enableCORS=false", "--server.enableXsrfProtection=false")
$sp = Start-Process -FilePath $py -ArgumentList $slArgs -WorkingDirectory $root `
        -RedirectStandardOutput $slog -RedirectStandardError "$slog.err" `
        -WindowStyle Hidden -PassThru
$sp.Id | Out-File -FilePath $pidFile -Append -Encoding ascii
"started {0,-18} port {1}  PID {2}" -f "streamlit-ui", 8501, $sp.Id

""
"All services launched. Logs in: $logs"
"Stop everything with:  .\run_services_local.ps1 -Stop"
