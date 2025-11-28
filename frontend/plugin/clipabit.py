import sys
import os
import requests
import uuid
import hashlib
import platform
from pathlib import Path

# Try to import PyQt6
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                                 QLabel, QPushButton, QMessageBox)
    from PyQt6.QtCore import Qt
except ImportError:
    print("Error: PyQt6 not found. Please run 'pip install PyQt6'")
    sys.exit(1)

# --- 1. Setup Resolve API ---
# We use a try-block so this script doesn't crash if you test it outside of Resolve
try:
    resolve = app.GetResolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    media_pool = project.GetMediaPool()
    project_timeline = project.GetCurrentTimeline()
except NameError:
    print("Warning: Resolve API not found. Running in simulation mode (external).")
    resolve, project, media_pool, = None, None, None
    
clip_map = {}
    
class ClipABitApp(QWidget):
    def __init__(self):
        super().__init__()
        
        clip_map = self._build_clip_map()  # Pre-build clip map on initialization
        print(clip_map)
        
        self.setWindowTitle("ClipABit Plugin (Resolve 20)")
        self.resize(300, 250)
        
        self.init_ui() 
        
    def init_ui(self):
        # Layout container (Vertical Box)
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. Label
        self.lbl_instruction = QLabel("<h3>ClipABit Actions</h3>")
        self.lbl_instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_instruction)

        # 3. Button: Append
        self.btn_append = QPushButton("Append Chunk to Timeline")
        self.btn_append.setMinimumHeight(40)
        self.btn_append.clicked.connect(self._action_append_chunk)
        layout.addWidget(self.btn_append)

        # Status Label (optional feedback)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.lbl_status)

        self.setLayout(layout)
    
    def _ensure_timeline(self):
        if not resolve:
            QMessageBox.warning(self, "Error", "Resolve API not available. Cannot perform action.")
            return False
        # Ask the project for the current timeline at runtime (don't rely on module-level cache)
        timeline = project.GetCurrentTimeline()
        if timeline:
            return True

        # No timeline: try to create an empty one and re-query
        create_ok = media_pool.CreateEmptyTimeline("New Timeline")
        if not create_ok:
            QMessageBox.warning(self, "Error", "Failed to create a new timeline.")
            return False

        # Re-query the project's current timeline
        timeline = project.GetCurrentTimeline()
        if not timeline:
            QMessageBox.warning(self, "Error", "Timeline creation reported success but no timeline found.")
            return False

        print("Created new empty timeline.")
        return True

    def _action_append_chunk(self):
        if not resolve:
            QMessageBox.warning(self, "Error", "Resolve API not available. Cannot perform action.")
            return
        
        print("Appending chunk to timeline...")
        
        timeline_exists = self._ensure_timeline()                
        if not timeline_exists:
            return

        # Get clips from the root folder (use Resolve API method names)
        root_folder = media_pool.GetRootFolder()
        clips = root_folder.GetClipList() if root_folder else []
        
        clips = list(filter(lambda c: "Video" in c.GetClipProperty("Type"), clips))
        if not clips:
            QMessageBox.information(self, "Info", "No clips found in media pool to append.")
            return

        # Append the first MediaPoolItem to the timeline. AppendToTimeline expects a list of MediaPoolItem objects.
        result = media_pool.AppendToTimeline([{"mediaPoolItem": clips[0], "startFrame": 0, "endFrame": 20}])
        if result:
            self.lbl_status.setText("Appended clip to timeline.")
            print("Appended clip to timeline.")
        else:
            QMessageBox.warning(self, "Error", "Failed to append clip to timeline.")

    def _extract_clip_fps(self, clip):
        """Try to read a clip's frame rate from common clip properties.

        Returns a float fps if found, otherwise None.
        """
        # Common property keys that Resolve may expose for frame rate
        keys = ("FPS", "Frame Rate", "FrameRate", "Video FPS", "Video Frame Rate")
        for k in keys:
            try:
                val = clip.GetClipProperty(k)
            except Exception:
                val = None
            if val:
                # Sometimes Resolve returns strings like '23.976' or '24'
                try:
                    return float(val)
                except Exception:
                    # Try to extract numeric portion
                    try:
                        num = ''.join(ch for ch in str(val) if (ch.isdigit() or ch == '.' or ch == ','))
                        num = num.replace(',', '.')
                        return float(num)
                    except Exception:
                        continue
        return None

    def _build_clip_map(self):
        """Scan media pool clips that match the current filter and build a mapping.

        The filter used is the same as in `_action_append_chunk`: keep clips
        where `"Video"` appears in `GetClipProperty("Type")`.

        Returns a dict mapping `filename` -> { 'media_pool_item': <item>, 'fps': <float|None> }
        If multiple items share the same filename, the value becomes a list of such dicts.
        """
        if not resolve:
            raise RuntimeError("Resolve API not available")

        root_folder = media_pool.GetRootFolder()
        clips = root_folder.GetClipList() if root_folder else []

        # Apply same filter as the append action
        filtered = [c for c in clips if c and c.GetClipProperty("Type") and "Video" in c.GetClipProperty("Type")]

        mapping = {}
        for clip in filtered:
            # Prefer File Path -> filename, fall back to clip name
            try:
                file_path = clip.GetClipProperty("File Path") or clip.GetClipProperty("FilePath")
            except Exception:
                file_path = None

            if file_path:
                filename = os.path.basename(file_path)
            else:
                # Some MediaPoolItem objects provide a Name
                try:
                    filename = clip.GetName() or "<unnamed>"
                except Exception:
                    filename = "<unnamed>"

            fps = self._extract_clip_fps(clip)

            entry = {"media_pool_item": clip, "fps": fps, "filepath": file_path}

            # Handle duplicate filenames by collecting into a list
            if filename in mapping:
                if isinstance(mapping[filename], list):
                    mapping[filename].append(entry)
                else:
                    mapping[filename] = [mapping[filename], entry]
            else:
                mapping[filename] = entry

        return mapping

    def _get_device_id_path(self) -> Path:
        """Return a platform-appropriate path to persist the device id."""
        system = platform.system()
        if system == "Windows":
            base = os.getenv("APPDATA") or str(Path.home())
            return Path(base) / "ClipABit" / "device_id.txt"
        elif system == "Darwin":
            return Path.home() / "Library" / "Application Support" / "ClipABit" / "device_id.txt"
        else:
            xdg = os.getenv("XDG_CONFIG_HOME")
            base = xdg if xdg else str(Path.home() / ".config")
            return Path(base) / "clipabit" / "device_id.txt"

    def get_or_create_device_id(self, persist: bool = True) -> str:
        """Get a persistent device id, create and store one if missing.

        If `persist` is False returns a generated id without saving.
        """
        path = self._get_device_id_path()

        try:
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except Exception:
            # ignore read errors and regenerate
            pass

        # Create a new id. Use uuid4 for randomness but prefix with a short host hash
        host = platform.node() or "unknown-host"
        host_hash = hashlib.sha1(host.encode("utf-8")).hexdigest()[:8]
        new_id = f"{host_hash}-{uuid.uuid4().hex}"

        if persist:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new_id, encoding="utf-8")
            except Exception:
                # If persisting fails, still return the generated id
                pass

        return new_id

    def get_project_name(self) -> str | None:
        """Return the current Resolve project name, or None if unavailable."""
        if not resolve or not project:
            return None

        # Try common method names; Resolve API varies by version
        for method in ("GetName", "GetProjectName", "GetTitle"):
            try:
                fn = getattr(project, method, None)
                if callable(fn):
                    name = fn()
                    if name:
                        return str(name)
            except Exception:
                continue

        # As a last resort try project manager lookup
        try:
            pm = resolve.GetProjectManager()
            cur = pm.GetCurrentProject()
            if cur:
                return getattr(cur, "GetName", lambda: None)()
        except Exception:
            pass

        return None

    def upload_files_for_embedding(self):
        for item in clip_map.items():
            project_name = self.get_project_name() 
            
           
        if not project_name:
                 QMessageBox.warning(self, "Error", "Unable to determine current project name, cannot upload files.")
                 return 
            
            device_id = self.get_or_create_device_id()
            
    def upload_file_to_backend(self, api_url: str, file_bytes: bytes, filename: str, content_type: str | None = None):
        """Upload file to backend via multipart form-data."""
        files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
        resp = requests.post(api_url, files=files, timeout=300)
        return resp            
            
        

# --- Main Execution ---



    app_qt = QApplication.instance()
    if not app_qt:
        app_qt = QApplication(sys.argv)
    
    window = ClipABitApp()
    window.show()
    window.raise_()
    window.activateWindow()
    
    print("ClipABit Plugin started.") 
    # Use exec() for PyQt6 (exec_ is deprecated)
    app_qt.exec()