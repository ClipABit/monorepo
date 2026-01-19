"""CLI for serving Modal apps locally."""

import signal
import subprocess
import sys
import threading

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
    """Serve all 3 Modal apps concurrently with color-coded prefixes."""
    processes = []
    threads = []
    
    def cleanup(signum=None, frame=None):
        """Handle Ctrl+C - terminate all processes."""
        print(f"\n{RESET}Shutting down all apps...")
        
        # First, try graceful termination
        for p in processes:
            try:
                p.terminate()
            except OSError:
                pass
        
        # Wait up to 3 seconds for graceful shutdown
        for p in processes:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                print(f"  Force killing process {p.pid}...")
                p.kill()
            except OSError:
                pass
        
        print("Done.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("Starting all Modal apps...\n")
    for name, (path, color) in APPS.items():
        print(f"  {color}●{RESET} {name}")
        p = subprocess.Popen(
            ["modal", "serve", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(p)
        
        # Start a thread to read and prefix output
        t = threading.Thread(target=_prefix_output, args=(p, name, color), daemon=True)
        t.start()
        threads.append(t)
    
    print(f"\nAll {len(APPS)} apps running. Press Ctrl+C to stop all.\n")
    print("-" * 60)
    
    # Wait for any process to exit (or Ctrl+C)
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        cleanup()


def serve_server():
    """Serve the API server app."""
    subprocess.run(["modal", "serve", APPS["server"][0]])


def serve_search():
    """Serve the search app."""
    subprocess.run(["modal", "serve", APPS["search"][0]])


def serve_processing():
    """Serve the processing app."""
    subprocess.run(["modal", "serve", APPS["processing"][0]])
