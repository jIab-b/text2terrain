#!/usr/bin/env python3
"""
Simple setup script for Text2Terrain using uv.
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: str, check: bool = True) -> int:
    """Run shell command and return exit code."""
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check).returncode

def main():
    """Set up Text2Terrain development environment."""
    
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✓ uv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing uv...")
        run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("✓ uv installed")
    
    # Create virtual environment
    print("Creating virtual environment...")
    run_cmd("uv venv")
    
    # Install core dependencies
    print("Installing core dependencies...")
    run_cmd("uv pip install -e .")
    
    # Ask user what extras to install
    print("\nOptional components:")
    print("1. Training dependencies (for Fireworks AI)")
    print("2. Development tools")
    print("3. C++ bindings (for Raylib renderer)")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4, or Enter to skip): ").strip()
    
    extras = []
    if choice in ["1", "4"]:
        extras.append("training")
    if choice in ["2", "4"]:
        extras.append("dev")
    if choice in ["3", "4"]:
        extras.append("bindings")
    
    if extras:
        extras_str = ",".join(extras)
        print(f"Installing extras: {extras_str}")
        run_cmd(f"uv pip install -e '[{extras_str}]'")
    
    print("\n🎉 Setup complete!")
    print("\nTo activate the environment:")
    print("  source .venv/bin/activate")
    print("\nTo get started:")
    print("  python -m src.data_gen.generator --help")

if __name__ == "__main__":
    main()