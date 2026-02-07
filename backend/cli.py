"""CLI for serving Modal apps locally."""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

# Dev combined app - all services in one for local iteration
DEV_COMBINED_APP = "apps/dev_combined.py"

# Individual apps for staging/prod deployment
APPS = {
    "server": ("apps/server.py", "\033[36m"),      # Cyan
    "search": ("apps/search_app.py", "\033[33m"),  # Yellow
    "processing": ("apps/processing_app.py", "\033[35m"),  # Magenta
}
RESET = "\033[0m"


def _prefix_output(process, name, color):
    """Read process output and prefix each line with the app name."""
    prefix = f"{color}[{name:^10}]{RESET} "
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                print(f"{prefix}{line}", end="", flush=True)
    except (ValueError, OSError):
        # Process closed, ignore
        pass


def serve_all():
    """
    Serve the combined dev app (all services in one).
    
    For local development, we use dev_combined.py which includes
    Server, Search, and Processing in a single Modal app.
    This allows hot-reload on all services without cross-app lookup issues.
    """
    print("Starting combined dev app (all services in one)...\n")
    print(f"  \033[32m●{RESET} dev-combined (server + search + processing)\n")
    print("Note: For staging/prod, deploy individual apps separately.\n")
    print("-" * 60 + "\n")
    
    # Ensure venv bin is in PATH for Modal subprocess calls
    venv_bin = Path(__file__).parent / ".venv" / "bin"
    env = os.environ.copy()
    if venv_bin.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}:{current_path}" if current_path else str(venv_bin)
    
    # Run with color-coded output prefixing
    color = "\033[32m"  # Green for combined dev app
    process = subprocess.Popen(
        ["uv", "run", "modal", "serve", DEV_COMBINED_APP],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stream output with color prefix
    _prefix_output(process, "dev", color)
    process.wait()


def _serve_single_app(name: str):
    """Serve a single app with color-coded output."""
    path, color = APPS[name]
    
    # Ensure venv bin is in PATH for Modal subprocess calls
    venv_bin = Path(__file__).parent / ".venv" / "bin"
    env = os.environ.copy()
    if venv_bin.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}:{current_path}" if current_path else str(venv_bin)
    
    process = subprocess.Popen(
        ["uv", "run", "modal", "serve", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stream output with color prefix
    _prefix_output(process, name, color)
    process.wait()


def serve_server():
    """Serve the API server app."""
    _serve_single_app("server")


def serve_search():
    """Serve the search app."""
    _serve_single_app("search")


def serve_processing():
    """Serve the processing app."""
    _serve_single_app("processing")


def serve_staging():
    """
    Serve all staging apps concurrently (server + search + processing).
    
    Runs all three apps in separate processes with ENVIRONMENT=staging.
    This matches the production architecture but runs locally.
    """
    print("Starting staging apps (all services separately)...\n")
    print(f"  \033[36m●{RESET} server\n")
    print(f"  \033[33m●{RESET} search\n")
    print(f"  \033[35m●{RESET} processing\n")
    print("Note: Cross-app communication works between these deployed apps.\n")
    print("-" * 60 + "\n")
    
    # Ensure venv bin is in PATH for Modal subprocess calls
    venv_bin = Path(__file__).parent / ".venv" / "bin"
    env = os.environ.copy()
    env["ENVIRONMENT"] = "staging"
    if venv_bin.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}:{current_path}" if current_path else str(venv_bin)
    
    # Start all three processes
    processes = []
    for name in ["server", "search", "processing"]:
        path, color = APPS[name]
        process = subprocess.Popen(
            ["uv", "run", "modal", "serve", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        processes.append((process, name, color))
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        for process, _, _ in processes:
            process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Stream output from all processes with color prefixing
    threads = []
    for process, name, color in processes:
        thread = threading.Thread(target=_prefix_output, args=(process, name, color))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all processes
    for process, _, _ in processes:
        process.wait()
