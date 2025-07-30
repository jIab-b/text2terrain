#!/usr/bin/env python3
"""
Master build script for Text2Terrain.

Handles all compilation and setup without CMake:
- Python package installation
- Optional dependency installation
- Raylib Python bindings setup
- Environment validation
"""

import subprocess
import sys
import argparse
from pathlib import Path
import platform


def run_cmd(cmd: str, check: bool = True, cwd: str = None) -> int:
    """Run shell command and return exit code."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    return result.returncode


def check_python_version():
    """Ensure Python version is compatible."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True


def check_uv_available() -> bool:
    """Check if uv is available."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✓ uv is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ uv not found, will use pip")
        return False


def install_dependencies(use_uv: bool = True, extras: list = None):
    """Install Python dependencies."""
    
    if extras is None:
        extras = []
    
    print("Installing dependencies...")
    
    if use_uv:
        # Install with uv
        if extras:
            extras_str = ",".join(extras)
            cmd = f"uv pip install -e '[{extras_str}]'"
        else:
            cmd = "uv pip install -e ."
    else:
        # Fallback to pip
        if extras:
            extras_str = ",".join(extras)
            cmd = f"pip install -e '[{extras_str}]'"
        else:
            cmd = "pip install -e ."
    
    return run_cmd(cmd)


def install_raylib():
    """Install Raylib Python bindings."""
    print("Installing Raylib Python bindings...")
    
    # Try different raylib packages
    raylib_packages = [
        "raylib-python",  # Main package
        "pyray",          # Alternative
    ]
    
    for package in raylib_packages:
        try:
            print(f"Trying to install {package}...")
            run_cmd(f"pip install {package}")
            
            # Test import
            result = subprocess.run([
                sys.executable, "-c", f"import {package.replace('-', '_')}; print('✓ {package} works')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ {package} installed successfully")
                return True
            else:
                print(f"✗ {package} import failed: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("⚠ Could not install Raylib. Terrain viewer will not work.")
    print("  You can try manually: pip install raylib-python")
    return False


def setup_development_environment():
    """Set up complete development environment."""
    
    print("Setting up development environment...")
    
    # Install development dependencies
    dev_packages = [
        "pytest",
        "black", 
        "jupyter",
        "matplotlib",  # For visualization
        "tqdm",        # Progress bars
    ]
    
    for package in dev_packages:
        try:
            run_cmd(f"pip install {package}", check=False)
        except:
            print(f"⚠ Failed to install {package}")


def validate_installation():
    """Validate that key components work."""
    
    print("Validating installation...")
    
    tests = [
        ("JAX", "import jax; print(f'JAX devices: {jax.devices()}')"),
        ("PyTorch", "import torch; print(f'PyTorch version: {torch.__version__}')"),
        ("Transformers", "import transformers; print(f'Transformers version: {transformers.__version__}')"),
        ("FastAPI", "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"),
        ("Procgen", "from src.procgen import TerrainEngine; print('✓ Procgen engine works')"),
    ]
    
    success_count = 0
    
    for name, test_code in tests:
        try:
            result = subprocess.run([
                sys.executable, "-c", test_code
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✓ {name}: {result.stdout.strip()}")
                success_count += 1
            else:
                print(f"✗ {name}: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ {name}: Timeout")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print(f"\nValidation complete: {success_count}/{len(tests)} components working")
    return success_count == len(tests)


def create_launch_scripts():
    """Create convenient launch scripts."""
    
    print("Creating launch scripts...")
    
    scripts = {
        "train.py": """#!/usr/bin/env python3
import sys
from src.training.train import main
if __name__ == "__main__":
    sys.exit(main())
""",
        "serve.py": """#!/usr/bin/env python3
import sys
from src.inference.api import main
if __name__ == "__main__":
    sys.exit(main())
""",
        "generate_data.py": """#!/usr/bin/env python3
import sys
from src.data_gen.generator import main
if __name__ == "__main__":
    sys.exit(main())
""",
        "viewer.py": """#!/usr/bin/env python3
import sys
from render.terrain_viewer import main
if __name__ == "__main__":
    sys.exit(main())
"""
    }
    
    for script_name, content in scripts.items():
        script_path = Path(script_name)
        with open(script_path, 'w') as f:
            f.write(content)
        
        # Make executable on Unix-like systems
        if platform.system() != "Windows":
            script_path.chmod(0o755)
        
        print(f"✓ Created {script_name}")


def print_next_steps():
    """Print instructions for next steps."""
    
    print("\n" + "="*60)
    print("🎉 Text2Terrain build complete!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Generate training data:")
    print("   python generate_data.py --n 10000 --output data/raw/demo")
    
    print("\n2. Preprocess data:")
    print("   python -m src.data_gen.preprocessing data/raw/demo/dataset_manifest.json --output data/processed")
    
    print("\n3. Train model locally (small dataset):")
    print("   python train.py --data-path data/processed --output-path models/demo --epochs 1")
    
    print("\n4. Train on Fireworks AI (large dataset):")
    print("   fws submit fireworks.yaml")
    
    print("\n5. Serve inference API:")
    print("   python serve.py --model-path models/demo/final_model.pt")
    
    print("\n6. Launch terrain viewer (if Raylib installed):")
    print("   python viewer.py --model-path models/demo/final_model.pt")
    
    print("\nDocs and examples:")
    print("  - API docs: http://localhost:8000/docs (when serving)")
    print("  - Notebooks: experiments/")
    print("  - Config files: configs/")


def main():
    """Main build script."""
    
    parser = argparse.ArgumentParser(description="Build Text2Terrain project")
    parser.add_argument("--minimal", action="store_true", help="Minimal installation (no dev tools)")
    parser.add_argument("--no-raylib", action="store_true", help="Skip Raylib installation")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation tests")
    parser.add_argument("--training", action="store_true", help="Install training dependencies")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    
    args = parser.parse_args()
    
    print("Text2Terrain Build Script")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Check for uv
    use_uv = check_uv_available()
    
    # Determine which extras to install
    extras = []
    if args.training:
        extras.append("training")
    if args.dev:
        extras.append("dev")
    
    try:
        # Install Python dependencies
        if install_dependencies(use_uv=use_uv, extras=extras) != 0:
            print("Failed to install Python dependencies")
            return 1
        
        # Install Raylib if requested
        if not args.no_raylib:
            install_raylib()
        
        # Set up development environment
        if not args.minimal:
            setup_development_environment()
        
        # Create launch scripts
        create_launch_scripts()
        
        # Validate installation
        if not args.no_validate:
            if not validate_installation():
                print("⚠ Some components failed validation")
        
        # Print next steps
        print_next_steps()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        return 1
    except Exception as e:
        print(f"Build failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())