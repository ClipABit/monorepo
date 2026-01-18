"""Watch `clipabit.py` and copy it to DaVinci Resolve Scripts/Utility on save.

Usage:
  python tools\watch_clipabit.py

Options / behavior:
- By default watches the workspace `frontend/plugin/clipabit.py` file and copies
  it into the target DaVinci Resolve Utility folder (path provided by the user).
- If `watchdog` is installed this uses an event-driven observer, otherwise falls
  back to a simple polling loop.

Run in PowerShell (from the repository root):

  python -m pip install watchdog  # optional but recommended
  python tools\watch_clipabit.py

You can stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
import platform
from pathlib import Path

logger = logging.getLogger("watch_clipabit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEFAULT_SOURCE = Path("../../frontend/plugin/ClipABit.py")
DEFAULT_COPY_NAME = "ClipABit.py"


def get_resolve_script_dir():
    """
    Returns the Path object for the DaVinci Resolve Utility Scripts directory
    based on the current operating system.
    """
    home = Path.home()
    system = platform.system()

    if system == "Windows":
        # Windows: %APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility
        # We use os.environ for APPDATA to be safer than hardcoding 'AppData/Roaming'
        base_path = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return base_path / "Blackmagic Design" / "DaVinci Resolve" / "Support" / "Fusion" / "Scripts" / "Utility"
        
    elif system == "Darwin":  # macOS
        # Mac: ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility
        return home / "Library" / "Application Support" / "Blackmagic Design" / "DaVinci Resolve" / "Fusion" / "Scripts" / "Utility"
    
    else:  # Linux
        # Linux: ~/.local/share/DaVinci Resolve/Fusion/Scripts/Utility
        # (Standard installation path for user-specific scripts)
        return home / ".local" / "share" / "DaVinci Resolve" / "Fusion" / "Scripts" / "Utility"   


def copy_file(src: Path, dst_dir: Path, dst_name: str = None) -> None:
    dst_dir = dst_dir.expanduser()
    if dst_name is None:
        dst_name = src.name

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name

    try:
        shutil.copy2(str(src), str(dst))
        logger.info("Copied %s -> %s", src, dst)
    except Exception as e:
        logger.error("Failed to copy %s -> %s: %s", src, dst, e)


def run_polling(src: Path, dst_dir: Path, dst_name: str = None, interval: float = 0.5):
    """Fallback polling loop if watchdog is not installed."""
    if not src.exists():
        logger.warning("Source file does not exist yet: %s", src)

    last_mtime = None
    try:
        while True:
            try:
                mtime = src.stat().st_mtime if src.exists() else None
            except Exception:
                mtime = None

            if mtime is not None and mtime != last_mtime:
                last_mtime = mtime
                copy_file(src, dst_dir, dst_name)

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Polling watcher stopped by user")


def run_watchdog(src: Path, dst_dir: Path, dst_name: str = None):
    """Use watchdog to observe the file and copy on modifications."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except Exception as e:
        logger.warning("watchdog not available (%s), falling back to polling", e)
        return run_polling(src, dst_dir, dst_name)

    class Handler(FileSystemEventHandler):
        def __init__(self, src_path: Path, dst_dir: Path, dst_name: str | None):
            super().__init__()
            self.src_path = src_path.resolve()
            self.dst_dir = dst_dir
            self.dst_name = dst_name
            self._last_copied = 0.0

        def on_modified(self, event):
            try:
                event_path = Path(event.src_path).resolve()
            except Exception:
                return

            # Debounce quick successive events (editors often trigger multiple events)
            now = time.time()
            if (now - self._last_copied) < 0.2:
                return

            if event_path == self.src_path:
                self._last_copied = now
                copy_file(self.src_path, self.dst_dir, self.dst_name)

        # Some editors replace files (moved/created) — copy on created too
        def on_created(self, event):
            try:
                event_path = Path(event.src_path).resolve()
            except Exception:
                return

            if event_path == self.src_path:
                copy_file(self.src_path, self.dst_dir, self.dst_name)

    handler = Handler(src.resolve(), dst_dir, dst_name)
    observer = Observer()

    # Watch the parent directory of the source file
    watch_dir = str(src.resolve().parent)
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    logger.info("Started watchdog observer for %s", src)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping observer")
        observer.stop()

    observer.join()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch ClipABit.py and copy to DaVinci Resolve Utility folder")
    p.add_argument("--source", "-s", type=str, default=str(DEFAULT_SOURCE), help="Path to local ClipABit.py (relative to repo root or absolute)")
    p.add_argument("--dest", "-d", type=str, default=str(get_resolve_script_dir()), help="Destination Utility folder")
    p.add_argument("--name", "-n", type=str, default=DEFAULT_COPY_NAME, help="Filename to write at destination")
    p.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval when watchdog unavailable")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source).expanduser()
    dst_dir = Path(args.dest).expanduser()

    # If source is relative, resolve relative to repository root (script location parent parent)
    if not src.is_absolute():
        # assume script is run from repo root; otherwise use script's parent
        repo_root = Path(__file__).resolve().parents[1]
        src = (repo_root / src).resolve()

    logger.info("Watching source: %s", src)
    logger.info("Destination directory: %s", dst_dir)

    # Copy once at startup so target is up-to-date before watching
    if src.exists():
        copy_file(src, dst_dir, args.name)
    else:
        logger.warning("Source file does not exist at startup: %s", src)

    # Try the event-driven watcher first
    try:
        run_watchdog(src, dst_dir, args.name)
    except Exception as e:
        logger.exception("Watchdog observer failed, falling back to polling: %s", e)
        run_polling(src, dst_dir, args.name, interval=args.poll_interval)


if __name__ == "__main__":
    main()
