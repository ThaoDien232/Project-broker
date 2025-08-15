@echo off
echo Syncing to GitHub...

:: Navigate to project directory
cd /d "d:\OneDrive - DRAGON CAPITAL\Thao Dien\Broker\Project broker"

:: Initialize git if needed
git init

:: Add remote if it doesn't exist
git remote remove origin 2>nul
git remote add origin https://github.com/ThaoDien232/Project-broker.git

:: Add all files
git add .

:: Commit changes
git commit -m "Update prop book dashboard with price fetching and profit/loss calculations"

:: Set main branch
git branch -M main

:: Push to GitHub
git push -u origin main

echo.
echo Sync complete! Check your GitHub repository.
pause
