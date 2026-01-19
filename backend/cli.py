"""CLI for serving Modal apps locally."""

import signal
import subprocess
import sys
import threading

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
    print(f"Starting combined dev app (all services in one)...\n")
    print(f"  \033[32m●{RESET} dev-combined (server + search + processing)\n")
    print(f"Note: For staging/prod, deploy individual apps separately.\n")
    print("-" * 60 + "\n")
    
    subprocess.run(["modal", "serve", DEV_COMBINED_APP])


def serve_server():
    """Serve the API server app."""
    subprocess.run(["modal", "serve", APPS["server"][0]])


def serve_search():
    """Serve the search app."""
    subprocess.run(["modal", "serve", APPS["search"][0]])


def serve_processing():
    """Serve the processing app."""
    subprocess.run(["modal", "serve", APPS["processing"][0]])
