"""
Automatic Rust extension builder for SPLICE.

Attempts to build the Rust-accelerated BAM reader. If any step fails,
falls back silently to the pure Python implementation.
"""

import os
import subprocess
import sys


def try_build_rust_extension():
    """Attempt to build the Rust BAM reader extension.

    Steps:
    1. Check if already installed
    2. Check/install Rust toolchain
    3. Check/install maturin
    4. Build with maturin develop --release

    If any step fails, prints a message and returns without error.
    The pure Python BAM reader produces identical results.
    """
    # Step 1: Check if already available
    try:
        import splice_rust  # noqa: F401
        print("Rust BAM reader: already installed")
        return
    except ImportError:
        pass

    print("Building Rust-accelerated BAM reader...")

    # Find project root (directory containing Cargo.toml)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cargo_toml = os.path.join(project_root, "Cargo.toml")
    if not os.path.exists(cargo_toml):
        print("  Cargo.toml not found. Skipping Rust build.")
        print("  Using Python BAM reader (slower but identical results).")
        return

    # Step 2: Check/install Rust
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise FileNotFoundError
        print(f"  Rust: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  Rust not found. Installing...")
        try:
            # Source cargo env in case it was just installed
            cargo_env = os.path.expanduser("~/.cargo/env")
            install_cmd = (
                'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs '
                '| sh -s -- -y'
            )
            subprocess.run(
                ["bash", "-c", install_cmd],
                capture_output=True, text=True, timeout=300,
            )
            # Update PATH
            cargo_bin = os.path.expanduser("~/.cargo/bin")
            if cargo_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = cargo_bin + ":" + os.environ.get("PATH", "")
            # Verify
            result = subprocess.run(
                ["rustc", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("Rust installation failed")
            print(f"  Rust installed: {result.stdout.strip()}")
        except Exception as e:
            print(f"  Could not install Rust: {e}")
            print("  Rust acceleration not available. Using Python BAM reader (slower but identical results).")
            return

    # Step 3: Check/install maturin
    try:
        import maturin  # noqa: F401
        print("  maturin: available")
    except ImportError:
        print("  Installing maturin...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "maturin"],
                capture_output=True, text=True, timeout=120,
            )
            print("  maturin: installed")
        except Exception as e:
            print(f"  Could not install maturin: {e}")
            print("  Rust acceleration not available. Using Python BAM reader (slower but identical results).")
            return

    # Step 4: Build with maturin
    print("  Compiling Rust extension (this may take a few minutes)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "develop", "--release"],
            capture_output=True, text=True, timeout=600,
            cwd=project_root,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:] if result.stderr else "Build failed")
        print("  Rust BAM reader: compiled and installed successfully")
    except Exception as e:
        print(f"  Rust build failed: {e}")
        print("  Rust acceleration not available. Using Python BAM reader (slower but identical results).")
        return

    # Verify
    try:
        import importlib
        if "splice_rust" in sys.modules:
            importlib.reload(sys.modules["splice_rust"])
        else:
            import splice_rust  # noqa: F401
        print("  Verification: Rust extension loaded successfully")
    except ImportError:
        print("  Warning: Build appeared to succeed but module not loadable.")
        print("  Using Python BAM reader (slower but identical results).")
