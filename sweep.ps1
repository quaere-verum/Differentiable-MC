# sweep.ps1
Set-Location $PSScriptRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "Sweep timestamp = $timestamp"
$hiddenSizesGrid = @(
    ,@()
    ,@(16)
    ,@(16, 16)
    ,@(16, 16, 16)
    ,@(16, 16, 16, 16)
    ,@(16, 16, 16, 16, 16)
)

$barrierGrid = @(80.0)
$transactionCostRateGrid = @(1e-3)
$volregime = @("normal", "stressed")
$lossName = @("mse", "cvar")
$varianceFeatureType = @(
    , "markov" 
    , "learned" 
    , "none"
    , "gated"
)

$runCount = 0

foreach ($hiddenSizes in $hiddenSizesGrid) {
    foreach ($barrier in $barrierGrid) {
        foreach ($transactionCostRate in $transactionCostRateGrid) {
            foreach ($vol in $volRegime) {
                foreach ($loss in $lossName) {
                    foreach ($vft in $varianceFeatureType) {
                        $runCount += 1

                        Write-Host ""
                        Write-Host "=================================================="
                        Write-Host "Run $runCount"
                        Write-Host "hidden_sizes          = $($hiddenSizes -join ',')"
                        Write-Host "barrier               = $barrier"
                        Write-Host "transaction_cost_rate = $transactionCostRate"
                        Write-Host "vol                   = $vol"
                        Write-Host "loss_name             = $loss"
                        Write-Host "variance_feature_type = $vft"
                        Write-Host "=================================================="

                        $cl_args = @(
                            "run",
                            "python",
                            "-m",
                            "scripts.barrier_hedging",
                            "--log-dir", "logs/$timestamp",
                            "--barrier", $barrier,
                            "--transaction-cost-rate", $transactionCostRate,
                            "--vol-regime", $vol,
                            "--risk-name", $loss,
                            "--variance-feature-type", $vft
                        )

                        if ($hiddenSizes.Count -gt 0) {
                            $cl_args += "--hidden-sizes"
                            foreach ($h in $hiddenSizes) {
                                $cl_args += $h
                            }
                        }

                        & uv @cl_args

                        if ($LASTEXITCODE -ne 0) {
                            Write-Error "Run $runCount failed with exit code $LASTEXITCODE."
                            exit $LASTEXITCODE
                        }
                    }
                }
            }
        }
    }
}

Write-Host ""
Write-Host "Completed $runCount runs successfully."