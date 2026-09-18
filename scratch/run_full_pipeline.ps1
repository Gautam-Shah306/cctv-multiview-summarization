$ErrorActionPreference = "Stop"

function Log-Step {
    param([string]$message)
    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$time] $message"
}

Log-Step "[STEP 1 START] Extracting ReID Features..."
python -m src.extract_reid_features --force --batch-size 64
if ($LASTEXITCODE -ne 0) { Log-Step "[ERROR] Step 1 failed with exit code $LASTEXITCODE."; exit $LASTEXITCODE }
Log-Step "[STEP 1 COMPLETE] ReID Extraction finished."

Log-Step "[VALIDATION START] Running similarity collapse check on view1..."
python scratch/validate_separation.py
if ($LASTEXITCODE -ne 0) { Log-Step "[ERROR] Validation failed. Collapse detected. Stopping."; exit $LASTEXITCODE }
Log-Step "[VALIDATION COMPLETE] Model output is sound."

Log-Step "[STEP 2 START] Pooling Tracklet Features..."
python -m src.pool_tracklet_features --force
if ($LASTEXITCODE -ne 0) { Log-Step "[ERROR] Step 2 failed with exit code $LASTEXITCODE."; exit $LASTEXITCODE }
Log-Step "[STEP 2 COMPLETE] Tracklet Pooling finished."

Log-Step "[STEP 3 START] Associating Cross-View Identities..."
python -m src.associate_cross_view
if ($LASTEXITCODE -ne 0) { Log-Step "[ERROR] Step 3 failed with exit code $LASTEXITCODE."; exit $LASTEXITCODE }
Log-Step "[STEP 3 COMPLETE] Cross-View Association finished."

Log-Step "[FILE TIMESTAMPS] Checking final output artifacts..."
$files = @(
    "data_storage\data\embeddings\view1_embeddings.npz",
    "data_storage\data\embeddings\view2_embeddings.npz",
    "data_storage\data\embeddings\view3_embeddings.npz",
    "data_storage\data\embeddings\tracklet_embeddings.npz",
    "outputs\global_identities.csv"
)
foreach ($f in $files) {
    if (Test-Path $f) {
        $item = Get-Item $f
        Write-Host ("{0,-60} {1}" -f $item.Name, $item.LastWriteTime)
    } else {
        Write-Host ("{0,-60} NOT FOUND" -f $f)
    }
}
Log-Step "[PIPELINE COMPLETE] All steps executed successfully."
