import os
import subprocess
import sys
from pathlib import Path


def serve() -> None:
    """
    Serve the Streamlit frontend.

    Usage: uv run dev <name>

    The name parameter is required in dev mode and must match the backend
    dev name to connect to the correct Modal app URLs.
    """
    env = os.environ.get("ENVIRONMENT", "dev")

    if env == "dev":
        if len(sys.argv) < 2:
            print("Error: Name parameter is required for dev mode.")
            print("Usage: uv run dev <name>")
            print("\nExample: uv run dev john")
            print("This connects to the Modal app named 'john-dev-server'")
            sys.exit(1)

        dev_name = sys.argv[1]
        os.environ["DEV_NAME"] = dev_name
        print(f"Starting frontend for dev instance '{dev_name}'...")

    app_path = Path(__file__).resolve().parent / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)], check=True)
