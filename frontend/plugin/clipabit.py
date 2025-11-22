import tkinter as tk
import json
import os

# 1. Setup the Resolve API link
resolve = app.GetResolve()
project_manager = resolve.GetProjectManager()
project = project_manager.GetCurrentProject()
media_pool = project.GetMediaPool()

# 2. Define where you want to save the JSON file
# CHANGE THIS PATH to your desired location.
# Use forward slashes '/' even on Windows to avoid errors.
OUTPUT_FILE = "C:/Users/sujas/Projects/monorepo/backend/media_pool_files.json"
# For Mac/Linux, it would look like: "/Users/yourname/Desktop/media_pool_files.json"

file_paths = []


def scan_folder(folder):
    """Recursively scans folders for clips."""

    # Get all clips in the current folder
    clips = folder.GetClipList()
    for clip in clips:
        # Get the file path property
        path = clip.GetClipProperty("File Path")

        # Only add if it has a valid path (ignores timelines, generators, etc.)
        if path:
            file_paths.append(path)

    # Recursively check subfolders
    subfolders = folder.GetSubFolderList()
    for subfolder in subfolders:
        scan_folder(subfolder)


# 3. Start the scan from the Root folder
root_folder = media_pool.GetRootFolder()
print("Scanning Media Pool...")
scan_folder(root_folder)


def on_button_click():
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(file_paths, f, indent=4)
        print(f"Success! Found {len(file_paths)} files.")
        print(f"Saved to: {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error saving file: {e}")
    print("Check that your directory exists and you have write permissions.")

    # You could put your file saving logic here later!


def upload_file_to_backend(api_url: str, file_bytes: bytes, filename: str, content_type: str | None = None):
    """Upload file to backend via multipart form-data."""
    files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
    resp = requests.post(api_url, files=files, timeout=300)
    return resp



def append_chunk_to_timeline():
    clips = root_folder.GetClipList()
    media_pool.AppendToTimeline(
        [{"mediaPoolItem": clips[0], "startFrame": 0, "endFrame": 20}])


# Setup
root = tk.Tk()
root.geometry("300x200")
root.title("ClipABit Plugin")

# Create a Label
lbl_instruction = tk.Label(root, text="Click below to run action:")
lbl_instruction.pack(pady=10)

# Create a Button
btn_action = tk.Button(
    root, text="Scan all media pool files", command=on_button_click)
btn_action2 = tk.Button(
    root, text="Append to timeline", command=append_chunk_to_timeline)

btn_action.pack(pady=5)
btn_action2.pack(pady=5)

# Start
root.mainloop()