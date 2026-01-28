"""CLI for serving Modal apps locally."""

import signal
import subprocess
import sys

# Dev combined app - all services in one for local iteration
DEV_COMBINED_APP = "apps/dev_combined.py"

# Individual apps for staging/prod deployment
APPS = {
    "server": ("services/http_server.py", "\033[36m"),      # Cyan
    "search": ("services/search_service.py", "\033[33m"),  # Yellow
    "processing": ("services/processing_service.py", "\033[35m"),  # Magenta
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
    
    # Run with color-coded output prefixing
    color = "\033[32m"  # Green for combined dev app
    process = subprocess.Popen(
        ["modal", "serve", DEV_COMBINED_APP],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
    
    process = subprocess.Popen(
        ["modal", "serve", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
