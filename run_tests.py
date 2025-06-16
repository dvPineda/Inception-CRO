#!/usr/bin/env python3
"""
Test runner script for Inception-CRO project.
Provides convenient commands for running different test suites.
"""

import subprocess
import sys
import argparse
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test runner for Inception-CRO")
    parser.add_argument('--fast', action='store_true', 
                       help='Run only fast tests (exclude slow/integration tests)')
    parser.add_argument('--smoke', action='store_true',
                       help='Run only smoke tests')
    parser.add_argument('--unit', action='store_true',
                       help='Run only unit tests (exclude integration)')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    
    args = parser.parse_args()
    
    # Ensure we're in the virtual environment
    if not os.path.exists('.venv'):
        print("Warning: No .venv directory found. Make sure you're using the virtual environment.")
    
    # Base pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    if args.verbose:
        cmd.append('-v')
    
    # Test selection
    if args.smoke:
        cmd.extend(['-k', 'smoke'])
        description = "Smoke Tests"
    elif args.unit:
        cmd.extend(['-m', 'not integration and not slow'])
        description = "Unit Tests"
    elif args.fast:
        cmd.extend(['-m', 'not slow'])
        description = "Fast Tests"
    else:
        description = "All Tests"
    
    # Coverage
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
        description += " with Coverage"
    
    # Run tests
    success = run_command(cmd, description)
    
    if args.coverage and success:
        print("\n📊 Coverage report generated in htmlcov/index.html")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

