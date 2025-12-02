#!/usr/bin/env python3
# check_dependencies.py - Verify all dependencies are installed

"""
Dependency checker for Inception-CRO project.
Run this script to verify all required packages are installed.
"""

import sys
import importlib
from typing import List, Tuple

# Required packages
REQUIRED_PACKAGES = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('sklearn', 'Scikit-learn'),
    ('pandas', 'Pandas'),
    ('matplotlib', 'Matplotlib'),
    ('psutil', 'PSUtil'),
    ('pytest', 'PyTest'),
]

# Optional packages
OPTIONAL_PACKAGES = [
    ('seaborn', 'Seaborn (recommended for better plots)'),
    ('medmnist', 'MedMNIST (for medical imaging datasets)'),
    ('jupyter', 'Jupyter (for notebook support)'),
    ('graphviz', 'Graphviz (for architecture visualization)'),
]

def check_package(package_name: str, description: str) -> Tuple[bool, str]:
    """
    Check if a package can be imported.
    
    Returns:
        (success, version_or_error)
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("🔍 Checking Inception-CRO Dependencies")
    print("=" * 50)
    
    required_ok = True
    
    # Check required packages
    print("\n📦 Required Packages:")
    for package, description in REQUIRED_PACKAGES:
        success, version_or_error = check_package(package, description)
        if success:
            print(f"  ✅ {description}: {version_or_error}")
        else:
            print(f"  ❌ {description}: MISSING")
            print(f"     Error: {version_or_error}")
            required_ok = False
    
    # Check optional packages
    print("\n📦 Optional Packages:")
    for package, description in OPTIONAL_PACKAGES:
        success, version_or_error = check_package(package, description)
        if success:
            print(f"  ✅ {description}: {version_or_error}")
        else:
            print(f"  ⚠️  {description}: Not installed")
    
    # Check Python version
    print(f"\n🐍 Python Version: {sys.version}")
    
    # Summary
    print("\n" + "=" * 50)
    if required_ok:
        print("✅ All required dependencies are available!")
        
        # Check for specific functionality
        print("\n🧪 Testing key functionality...")
        
        # Test PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("  ℹ️  CUDA not available (CPU only)")
        except Exception as e:
            print(f"  ⚠️  PyTorch test failed: {e}")
        
        # Test Seaborn
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.style.use('default')  # Test style setting
            print("  ✅ Matplotlib and Seaborn integration working")
        except Exception as e:
            print(f"  ⚠️  Visualization test warning: {e}")
        
        # Test sklearn
        try:
            from sklearn.metrics import accuracy_score
            print("  ✅ Scikit-learn metrics available")
        except Exception as e:
            print(f"  ⚠️  Scikit-learn test failed: {e}")
        
        print("\n🚀 Ready to run Inception-CRO!")
        print("\nNext steps:")
        print("  1. Run demo: ./inception_cro demo")
        print("  2. Train model: ./inception_cro train")
        print("  3. Compare models: ./inception_cro compare --mode quick")
        
    else:
        print("❌ Some required dependencies are missing!")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

