$guid = "{551BF148-7B62-436A-8A4F-9C1D1E2F3A4B}"
$zipName = "$guid.zip"
$files = "manifest.json", "index.html", "code.js", "webcc.min.js", "assets"

Write-Host "checking for files..."

# Check metadata
if (-not (Test-Path "webcc.min.js")) {
    Write-Warning "webcc.min.js is MISSING."
    Write-Warning "You must copy webcc.min.js from a Siemens example project (Entry ID: 109779176) into this folder."
    Write-Host "Cannot proceed without webcc.min.js." -ForegroundColor Red
    exit
}

$webccContent = Get-Content "webcc.min.js" -Raw
if ($webccContent -match "PLACEHOLDER") {
    Write-Warning "webcc.min.js is still the PLACEHOLDER file."
    Write-Warning "You need to replace it with the actual Siemens library file."
    exit
}

Write-Host "All files present. Creating package..."

# Remove old zip if exists
if (Test-Path $zipName) { Remove-Item $zipName }

# Create Zip
Compress-Archive -Path $files -DestinationPath $zipName -Force

Write-Host "---------------------------------------------------" -ForegroundColor Green
Write-Host "SUCCESS! Created Custom Web Control Package:" -ForegroundColor Green
Write-Host "$PWD\$zipName" -ForegroundColor Cyan
Write-Host "---------------------------------------------------" -ForegroundColor Green
Write-Host "NEXT STEP: "
Write-Host "1. Open your TIA Portal Project folder in Windows Explorer."
Write-Host "2. Go to: \UserFiles\CustomControls\"
Write-Host "   (If the folder doesn't exist, create it inside your project folder)"
Write-Host "3. Copy the .zip file created above into that folder."
Write-Host "4. Open TIA Portal -> Toolbox -> My Controls -> Hit 'Refresh' button."
