param([string]$Root = (Split-Path -Parent $PSScriptRoot), [string]$Python = 'python')
param([switch]$Smoke, [double]$Epochs = 3, [int]$MaxLen = 256, [double]$MaxMinutes = 200)
# Cadence Flash phase 2 launcher. Whole job (train+eval) is killed at $MaxMinutes; training itself stops at MaxMinutes-30 to leave eval time.
$G = $Root; Set-Location $G
$liveName = if ($Smoke) { 'p2\live-smoke' } else { 'p2\live' }; $LIVE = "$G\$liveName"; New-Item -ItemType Directory -Force $LIVE | Out-Null
function L($m) { $s = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $m; Add-Content -Path "$LIVE\launcher.log" -Value $s }
L "launcher start pid=$PID smoke=$Smoke epochs=$Epochs maxlen=$MaxLen maxminutes=$MaxMinutes"
if (Test-Path "$G\STOP") { L "STOP file present at start; exiting"; exit 0 }
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'standard_bench|pilot\.py|train\.py|train_p2\.py|rescore' }
if ($busy) { L "another GPU python job running ($($busy.ProcessId -join ',')); exiting"; exit 0 }
function Reload { $body = '{"model":"qwen2.5:3b-instruct","prompt":"","keep_alive":-1,"options":{"num_ctx":2048}}'
  try { Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 300 | Out-Null; L "qwen2.5 reloaded"; L ("ollama ps: " + ((& ollama ps 2>&1 | Select-Object -Skip 1) -join ' / ')) } catch { L "qwen2.5 reload FAILED: $($_.Exception.Message)" } }
try {
  foreach ($m in 'qwen2.5:3b-instruct','phi4-mini') { & ollama stop $m 2>&1 | Out-Null; L "ollama stop $m exit=$LASTEXITCODE" }
  Start-Sleep 3; $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); L "gpu used after unload: $used MiB"
  if ($used -gt 4000) { L "GPU not free (>4000 MiB used); aborting"; throw "gpu busy" }
  foreach ($v in 'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','HF_API_TOKEN','PYTHONPATH','PYTHONHOME','CUDA_VISIBLE_DEVICES') { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
  $env:HF_HOME = "$G\hf-home"; $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'; $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'; $env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
  $py = $Python
  $trainMin = [Math]::Max(1, $MaxMinutes - 30)
  $args2 = @('-u','train_p2.py','--live',$liveName,'--epochs',"$Epochs",'--max_len',"$MaxLen",'--max_minutes',"$trainMin")
  if ($Smoke) { $args2 += @('--smoke_seconds','30','--eval_cap','16') }
  $p = Start-Process -FilePath $py -ArgumentList $args2 -WorkingDirectory $G -WindowStyle Hidden -PassThru -RedirectStandardOutput "$LIVE\train.out.log" -RedirectStandardError "$LIVE\train.err.log"
  $null = $p.Handle; L "train pid=$($p.Id) train_budget_min=$trainMin"; $net = 0; $t0 = Get-Date
  while (-not $p.HasExited) {
    Start-Sleep 15
    $ids = @($p.Id) + @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" | % ProcessId)
    $c = Get-NetTCPConnection -ErrorAction SilentlyContinue | Where-Object { $ids -contains $_.OwningProcess -and $_.RemoteAddress -notin '127.0.0.1','::1','0.0.0.0','::' }
    if ($c) { $net += @($c).Count; L ("NET connection: " + (($c | % { "$($_.RemoteAddress):$($_.RemotePort)" }) -join ',')) }
    if ((Get-Date).Second -lt 15) { L ("gpu: " + (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)) }
    if (((Get-Date) - $t0).TotalMinutes -gt $MaxMinutes) { L "200-min guard hit; killing train pid=$($p.Id)"; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; break }
  }
  L "train exit=$($p.ExitCode) net_connections=$net"
} catch { L "launcher error: $($_.Exception.Message)" }
finally { Reload; L "QUEUE END cadence-flash-p2" }
