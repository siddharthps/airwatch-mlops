# PowerShell Script: run_full_training_and_selection.ps1

# Ensure we are in the flows directory
Set-Location -Path $PSScriptRoot

# Start MLflow server in background
Write-Host "Starting MLflow tracking server..."
# --- CHANGE MADE HERE: Updated --default-artifact-root ---
$mlflowProc = Start-Process `
    -FilePath "mlflow" `
    -ArgumentList "server", "--backend-store-uri", "sqlite:///../mlflow.db", "--default-artifact-root", "s3://mlflow-artifacts-chicago-2025/", "--host", "127.0.0.1", "--port", "5000" `
    -PassThru -NoNewWindow # -NoNewWindow will keep it in the same terminal for easier cleanup

# Wait for MLflow to start
Write-Host "Waiting for MLflow server to start..."
Start-Sleep -Seconds 10

# Run model training flow
Write-Host "`nRunning model training flow..."
python .\model_training.py

# Wait for training logs to flush
Write-Host "Waiting for training logs to flush..."
Start-Sleep -Seconds 5

# Run model selection + S3 upload
Write-Host "`nRunning model selector and summary uploader..."
$env:PREFECT_LOGGING_LEVEL = "INFO" # Ensure Prefect logs are visible
python .\model_selector.py

# Stop MLflow server
Write-Host "`nStopping MLflow tracking server..."
if ($mlflowProc -and !$mlflowProc.HasExited) {
    # Check if the process is still running and then terminate it
    Stop-Process -Id $mlflowProc.Id -Force -ErrorAction SilentlyContinue
    Write-Host "MLflow server stopped."
} elseif ($mlflowProc -and $mlflowProc.HasExited) {
    Write-Host "MLflow server exited before manual stop (possibly due to error or already stopped)."
} else {
    Write-Host "MLflow process object not found or failed to start."
}