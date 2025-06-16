import subprocess
import sys
import os
import tempfile
import shutil
import pytest
from pathlib import Path

@pytest.mark.timeout(300)  # allow a few minutes if GPU is slow
def test_main_runs_briefly(tmp_path):
    """
    Test that main.py runs successfully with minimal parameters for quick execution.
    Uses actual CLI arguments to override default config.
    """
    # Create a temporary experiments directory
    exp_dir = tmp_path / "test_experiments"
    exp_dir.mkdir()
    
    # 1. Build command with minimal parameters for fast execution
    cmd = [
        sys.executable, "-u", "main.py",
        "--experiments-dir", str(exp_dir),
        "--reef-size", "(2, 2)",
        "--max-generations", "1", 
        "--max-no-improve", "1",
        "--branch-min", "1",
        "--branch-max", "1", 
        "--batch-size", "4",
        "--num-batches", "1",
        "--num-epochs", "1",
        "--patience", "1",
        "--fitness-alpha", "3",
        "--seed", "42",
        "--no-shuffle-dataset"  # Faster without shuffling
    ]
    
    # Ensure current working directory is project root
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, "main.py")):
        pytest.skip("main.py not found in current working directory")
    
    # 2. Run the command
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail("main.py execution timed out after 300 seconds")
    except FileNotFoundError:
        pytest.skip("Python executable or main.py not found")
    
    # 3. Check return code and output
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"main.py failed with return code {result.returncode}")
    
    # 4. Verify experiment structure was created
    exp_folders = list(exp_dir.glob("*"))
    assert len(exp_folders) > 0, "No experiment folders created"
    
    # 5. Check for expected output files
    date_folder = exp_folders[0]
    if date_folder.is_dir():
        param_folders = list(date_folder.glob("*"))
        if param_folders:
            exp_folder = param_folders[0]
            if exp_folder.is_dir():
                exp_run_folders = list(exp_folder.glob("exp_*"))
                if exp_run_folders:
                    run_folder = exp_run_folders[0]
                    # Check for essential files
                    expected_files = ["results.csv", "config_used.json"]
                    for expected_file in expected_files:
                        file_path = run_folder / expected_file
                        assert file_path.exists(), f"Expected file {expected_file} not found in {run_folder}"

def test_main_help_works():
    """
    Test that main.py --help works correctly.
    """
    cmd = [sys.executable, "main.py", "--help"]
    cwd = os.getcwd()
    
    if not os.path.exists(os.path.join(cwd, "main.py")):
        pytest.skip("main.py not found in current working directory")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("main.py --help timed out")
    except FileNotFoundError:
        pytest.skip("Python executable or main.py not found")
    
    # Help should exit with code 0 and contain usage info
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "Inception-CRO" in result.stdout
