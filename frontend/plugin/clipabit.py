import sys
import os
import requests
import uuid
import hashlib
import platform
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Try to import PyQt6
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                 QLabel, QPushButton, QMessageBox, QLineEdit,
                                 QScrollArea, QFrame, QSplitter, QListWidget, QListWidgetItem,
                                 QDialog, QCheckBox)
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
except ImportError:
    print("Error: PyQt6 not found. Please run 'pip install PyQt6'")
    sys.exit(1)

# --- Configuration ---
class Config:
    """Configuration for ClipABit plugin."""
    
    # Environment (can be overridden via environment variable)
    ENVIRONMENT = os.environ.get("CLIPABIT_ENVIRONMENT", "dev")
    
    # API Endpoints - dynamically constructed based on environment
    url_portion = "" if ENVIRONMENT in ["prod", "staging"] else f"-{ENVIRONMENT}"
    
    SEARCH_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-search{url_portion}.modal.run"
    UPLOAD_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-upload{url_portion}.modal.run"
    STATUS_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-status{url_portion}.modal.run"
    DELETE_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-delete-video{url_portion}.modal.run"
    CHECK_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-check-video{url_portion}.modal.run"
    
    
    # Timeouts and delays
    UPLOAD_TIMEOUT = 300
    STATUS_CHECK_TIMEOUT = 10
    STATUS_CHECK_INTERVAL = 2
    QUEUE_DELAY = 1000  # milliseconds
    

# --- Background Job Tracker Thread ---
class JobTracker(QThread):
    """Background thread to track upload job status."""
    
    job_completed = pyqtSignal(str, dict)  # job_id, result
    job_failed = pyqtSignal(str, str)      # job_id, error
    
    def __init__(self):
        super().__init__()
        self.jobs_to_track = {}  # job_id -> job_info
        self.running = True
        
    def add_job(self, job_id: str, job_info: dict):
        """Add a job to track."""
        self.jobs_to_track[job_id] = job_info
        
    def run(self):
        """Main tracking loop."""
        while self.running:
            jobs_to_remove = []
            
            for job_id, job_info in self.jobs_to_track.items():
                try:
                    response = requests.get(Config.STATUS_API_URL, params={"job_id": job_id}, timeout=Config.STATUS_CHECK_TIMEOUT)
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status", "processing")
                        
                        if status == "completed":
                            self.job_completed.emit(job_id, data)
                            jobs_to_remove.append(job_id)
                        elif status == "failed":
                            error = data.get("error", "Unknown error")
                            self.job_failed.emit(job_id, error)
                            jobs_to_remove.append(job_id)
                        
                except Exception as e:
                    error_msg = f"Error checking job {job_id}: {e}\n{traceback.format_exc()}"
                    print(error_msg)
                    
            # Remove completed/failed jobs
            for job_id in jobs_to_remove:
                del self.jobs_to_track[job_id]
                
            time.sleep(Config.STATUS_CHECK_INTERVAL)
            
    def stop(self):
        """Stop the tracking thread."""
        self.running = False
    
# --- Setup Resolve API ---
# We use a try-block so this script doesn't crash if you test it outside of Resolve
try:
    resolve = app.GetResolve()  # type: ignore  # app is provided by Resolve environment
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    media_pool = project.GetMediaPool()
except NameError:
    print("Warning: Resolve API not found. Running in simulation mode (external).")
    resolve, project, media_pool, = None, None, None

class ClipABitApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize data
        self.clip_map = {}
        self.processed_files = self._load_processed_files()
        self.current_jobs = {}  # job_id -> job_info
        
        # Upload queue system
        self.upload_queue = []  # List of files waiting to be uploaded
        self.current_upload = None  # Currently uploading file info
        self.is_uploading = False  # Flag to prevent concurrent uploads
        
        # Initialize job tracker
        self.job_tracker = JobTracker()
        self.job_tracker.job_completed.connect(self._on_job_completed)
        self.job_tracker.job_failed.connect(self._on_job_failed)
        self.job_tracker.start()
        
        # Build clip map and check for new files (only if Resolve is available)
        if resolve:
            self.clip_map = self._build_clip_map(debug=False)
            print(f"Found {len(self.clip_map)} clips in media pool")
        else:
            self.clip_map = {}
            print("Running without Resolve API - clip map disabled")
        
        # Setup UI
        self.setWindowTitle("ClipABit Plugin (Resolve 20)")
        self.resize(800, 600)
        self.init_ui()
        
        # Setup refresh timer (disabled by default; refresh on demand)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(lambda: self._refresh_media_pool(debug=False))
        
        # Run consistency check on startup
        self._run_consistency_check("startup")
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("<h2>ClipABit - Semantic Video Search</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Upload and Jobs
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Search and Results
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 10px; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
    def _create_left_panel(self):
        """Create the left panel with upload and job tracking."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Upload section
        upload_frame = QFrame()
        upload_frame.setFrameStyle(QFrame.Shape.Box)
        upload_layout = QVBoxLayout()
        
        upload_title = QLabel("<b>Upload Media</b>")
        upload_layout.addWidget(upload_title)
        
        # Upload button
        self.btn_select_files = QPushButton("Select Files to Upload")
        self.btn_select_files.setMinimumHeight(35)
        self.btn_select_files.clicked.connect(self._select_files_to_upload)
        upload_layout.addWidget(self.btn_select_files)
        
        # Clear queue button
        self.btn_clear_queue = QPushButton("Clear Queue")
        self.btn_clear_queue.setMinimumHeight(30)
        self.btn_clear_queue.clicked.connect(self._clear_upload_queue)
        self.btn_clear_queue.setStyleSheet("background-color: #ff6b6b; color: white;")
        upload_layout.addWidget(self.btn_clear_queue)
        
        # File status
        self.file_status_label = QLabel("Checking media pool...")
        upload_layout.addWidget(self.file_status_label)
        
        upload_frame.setLayout(upload_layout)
        layout.addWidget(upload_frame)
        
        # Jobs section
        jobs_frame = QFrame()
        jobs_frame.setFrameStyle(QFrame.Shape.Box)
        jobs_layout = QVBoxLayout()
        
        jobs_title = QLabel("<b>Active Jobs</b>")
        jobs_layout.addWidget(jobs_title)
        
        # Jobs list
        self.jobs_list = QListWidget()
        self.jobs_list.setMaximumHeight(200)
        jobs_layout.addWidget(self.jobs_list)
        
        jobs_frame.setLayout(jobs_layout)
        layout.addWidget(jobs_frame)
        
        # Debug/Testing section
        debug_frame = QFrame()
        debug_frame.setFrameStyle(QFrame.Shape.Box)
        debug_layout = QVBoxLayout()
        
        debug_title = QLabel("<b>Debug Info</b>")
        debug_layout.addWidget(debug_title)
        
        # Storage path
        storage_path = self._get_storage_path()
        self.storage_path_label = QLabel(f"Storage: {storage_path}")
        self.storage_path_label.setWordWrap(True)
        self.storage_path_label.setStyleSheet("color: gray; font-size: 9px;")
        debug_layout.addWidget(self.storage_path_label)
        
        # Processed files count
        self.processed_count_label = QLabel("Processed: 0 files")
        self.processed_count_label.setStyleSheet("color: gray; font-size: 9px;")
        debug_layout.addWidget(self.processed_count_label)
        
        # View processed files button
        btn_view_processed = QPushButton("View Processed Files")
        btn_view_processed.setMinimumHeight(25)
        btn_view_processed.clicked.connect(self._show_processed_files)
        btn_view_processed.setStyleSheet("font-size: 9px;")
        debug_layout.addWidget(btn_view_processed)

        # Verify backend button
        btn_verify_backend = QPushButton("Verify Backend")
        btn_verify_backend.setMinimumHeight(25)
        btn_verify_backend.clicked.connect(self._verify_backend_records)
        btn_verify_backend.setStyleSheet("font-size: 9px; background-color: #6c8cff; color: white;")
        debug_layout.addWidget(btn_verify_backend)
        
        # Clear processed files button
        btn_clear_processed = QPushButton("Clear Processed Files")
        btn_clear_processed.setMinimumHeight(25)
        btn_clear_processed.clicked.connect(self._clear_processed_files)
        btn_clear_processed.setStyleSheet("font-size: 9px; background-color: #ffaa00; color: white;")
        debug_layout.addWidget(btn_clear_processed)
        
        debug_frame.setLayout(debug_layout)
        layout.addWidget(debug_frame)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def _create_right_panel(self):
        """Create the right panel with search and results."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Search section
        search_frame = QFrame()
        search_frame.setFrameStyle(QFrame.Shape.Box)
        search_layout = QVBoxLayout()
        
        search_title = QLabel("<b>Search Videos</b>")
        search_layout.addWidget(search_title)
        
        # Search input
        search_input_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query (e.g., 'woman walking', 'car driving')")
        self.search_input.returnPressed.connect(self._perform_search)
        search_input_layout.addWidget(self.search_input)
        
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self._perform_search)
        search_input_layout.addWidget(self.btn_search)
        
        search_layout.addLayout(search_input_layout)
        search_frame.setLayout(search_layout)
        layout.addWidget(search_frame)
        
        # Results section
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.Shape.Box)
        results_layout = QVBoxLayout()
        
        results_title = QLabel("<b>Search Results</b>")
        results_layout.addWidget(results_title)
        
        # Results scroll area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_scroll.setWidget(self.results_widget)
        
        results_layout.addWidget(self.results_scroll)
        results_frame.setLayout(results_layout)
        layout.addWidget(results_frame)
        
        panel.setLayout(layout)
        return panel
    def _get_storage_path(self) -> Path:
        """Get path for local storage."""
        try:
            script_path = Path(__file__).resolve()
        except Exception:
            script_arg = sys.argv[0] if len(sys.argv) > 0 else ""
            if script_arg:
                script_path = Path(script_arg)
                if not script_path.is_absolute():
                    script_path = (Path.cwd() / script_path).resolve()
            else:
                script_path = Path.cwd()
        return script_path.parent / "processed_files.json"
        
    def _load_processed_files(self) -> Dict[str, Dict]:
        """Load list of processed files from local storage."""
        storage_path = self._get_storage_path()
        try:
            if storage_path.exists():
                with open(storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading processed files: {e}")
        return {}
        
    def _save_processed_files(self):
        """Save processed files to local storage."""
        storage_path = self._get_storage_path()
        try:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(storage_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            print(f"Error saving processed files: {e}")
            
    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash for file to track changes."""
        try:
            stat = os.stat(filepath)
            # Use file path, size, and modification time for hash
            content = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return hashlib.md5(filepath.encode()).hexdigest()

    def _get_hashed_identifier(self, filepath: str, namespace: str, filename: str) -> str:
        """Match backend identifier generation for plugin uploads."""
        identifier_source = filepath if filepath else f"{namespace}/{filename}"
        return hashlib.sha256(identifier_source.encode()).hexdigest()
            
    def _refresh_media_pool(self, debug: bool = False):
        """Refresh media pool and update file status."""
        if not resolve:
            return
            
        self.clip_map = self._build_clip_map(debug=debug)
        if debug:
            print(f"[MediaPool] Refreshed clip map: {len(self.clip_map)} unique filenames")
        self._update_file_status()
        
    def _clear_upload_queue(self):
        """Clear the upload queue."""
        if not self.upload_queue and not self.is_uploading:
            QMessageBox.information(self, "Info", "No uploads to cancel.")
            return
            
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 
            "Clear Upload Queue", 
            f"Cancel {len(self.upload_queue)} queued uploads?\n\nCurrently uploading file will continue.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            cleared_count = len(self.upload_queue)
            
            # Remove queued jobs from display
            jobs_to_remove = []
            for job_id, job_info in self.current_jobs.items():
                if job_info.get('status') == 'queued':
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.current_jobs[job_id]
            
            # Clear the queue
            self.upload_queue.clear()
            
            # Update displays
            self._update_jobs_display()
            self.status_label.setText(f"Cleared {cleared_count} uploads from queue")
            self._update_file_status()
            
    def _update_file_status(self):
        """Update the file status display."""
        total_files = len(self.clip_map)
        processed_count = 0
        new_files = []
        
        for filename, clip_info in self.clip_map.items():
            if isinstance(clip_info, list):
                # Handle multiple clips with same filename
                for clip in clip_info:
                    filepath = clip.get('filepath')
                    if filepath:
                        file_hash = self._get_file_hash(filepath)
                        if file_hash in self.processed_files:
                            processed_count += 1
                        else:
                            new_files.append(filename)
            else:
                filepath = clip_info.get('filepath')
                if filepath:
                    file_hash = self._get_file_hash(filepath)
                    if file_hash in self.processed_files:
                        processed_count += 1
                    else:
                        new_files.append(filename)
        
        new_count = len(set(new_files))
        queued_count = len(self.upload_queue)
        
        # Update status text to include queue info
        status_parts = [f"Files: {total_files} total, {processed_count} processed, {new_count} new"]
        if queued_count > 0:
            status_parts.append(f"{queued_count} queued")
        if self.is_uploading:
            status_parts.append("uploading...")
            
        status_text = ", ".join(status_parts)
        self.file_status_label.setText(status_text)
        
        # Update processed files count in debug section
        if hasattr(self, 'processed_count_label'):
            total_processed = len(self.processed_files)
            self.processed_count_label.setText(f"Processed: {total_processed} files")
        
        # Update button states - disable during upload
        self.btn_select_files.setEnabled(not self.is_uploading)
        self.btn_clear_queue.setEnabled(queued_count > 0 or self.is_uploading)
        
    def _select_files_to_upload(self):
        """Select media pool clips and add them to the upload queue."""
        if not resolve:
            QMessageBox.warning(self, "Resolve Not Available", "Resolve API is not available. Media pool selection is disabled.")
            return

        # Refresh media pool so we show the latest clips
        self._refresh_media_pool(debug=False)

        files_to_upload = []
        skipped_processed = 0
        skipped_queued = 0
        skipped_missing = 0
        seen_paths = set()

        def collect_candidate(entry):
            nonlocal skipped_processed, skipped_queued, skipped_missing
            filepath = entry.get('filepath')
            if not filepath:
                skipped_missing += 1
                return
            if filepath in seen_paths:
                return
            seen_paths.add(filepath)

            if not os.path.exists(filepath):
                skipped_missing += 1
                return

            filename = os.path.basename(filepath)
            file_hash = self._get_file_hash(filepath)

            if file_hash in self.processed_files:
                skipped_processed += 1
                return

            if self._is_file_being_processed(file_hash):
                skipped_queued += 1
                return

            files_to_upload.append({
                'filepath': filepath,
                'filename': filename,
                'hash': file_hash
            })

        for _, clip_info in self.clip_map.items():
            if isinstance(clip_info, list):
                for entry in clip_info:
                    collect_candidate(entry)
            else:
                collect_candidate(clip_info)

        if not files_to_upload:
            QMessageBox.information(
                self,
                "Info",
                f"No eligible media pool clips to upload.\nProcessed: {skipped_processed}, In queue: {skipped_queued}, Missing path: {skipped_missing}"
            )
            return

        # Build selection dialog from media pool candidates
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Media Pool Clips")
        dialog.resize(700, 500)
        dialog.setModal(True)

        layout = QVBoxLayout()
        header = QLabel(f"<b>Select clips to upload ({len(files_to_upload)} available)</b>")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        checkboxes = []
        for entry in files_to_upload:
            filename = entry.get('filename', 'Unknown')
            filepath = entry.get('filepath', '')
            checkbox = QCheckBox(f"{filename}\n{filepath}")
            checkbox.setChecked(False)
            checkbox.setToolTip(filepath)
            checkbox.setStyleSheet("font-size: 11px;")
            scroll_layout.addWidget(checkbox)
            checkboxes.append((checkbox, entry))

        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons
        button_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_none = QPushButton("Select None")
        btn_cancel = QPushButton("Cancel")
        btn_add = QPushButton("Add Selected")

        def set_all(state: bool):
            for cb, _ in checkboxes:
                cb.setChecked(state)

        btn_select_all.clicked.connect(lambda: set_all(True))
        btn_select_none.clicked.connect(lambda: set_all(False))
        btn_cancel.clicked.connect(dialog.reject)
        btn_add.clicked.connect(dialog.accept)

        button_row.addWidget(btn_select_all)
        button_row.addWidget(btn_select_none)
        button_row.addStretch()
        button_row.addWidget(btn_cancel)
        button_row.addWidget(btn_add)

        layout.addLayout(button_row)
        dialog.setLayout(layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_files = [entry for cb, entry in checkboxes if cb.isChecked()]

        if not selected_files:
            QMessageBox.information(self, "Info", "No clips selected for upload.")
            return

        # Add files to upload queue
        self.upload_queue.extend(selected_files)

        # Create temporary job entries for UI
        for file_info in selected_files:
            temp_job_id = f"queued_{file_info['hash'][:8]}"
            self.current_jobs[temp_job_id] = {
                'filename': file_info['filename'],
                'filepath': file_info['filepath'],
                'file_hash': file_info['hash'],
                'status': 'queued',
                'temp_job': True
            }

        self._update_jobs_display()
        self._update_file_status()

        # Start processing queue
        if not self.is_uploading:
            self._process_upload_queue()

        QMessageBox.information(
            self,
            "Queued",
            f"Added {len(selected_files)} file(s) to upload queue.\nSkipped processed: {skipped_processed}, in queue: {skipped_queued}, missing: {skipped_missing}"
        )
            
    def _is_file_being_processed(self, file_hash: str) -> bool:
        """Check if file is currently being processed."""
        # Check if file is in current jobs (being processed)
        for job_info in self.current_jobs.values():
            if job_info.get('file_hash') == file_hash:
                return True
        
        # Check if file is in upload queue
        for file_info in self.upload_queue:
            if file_info.get('hash') == file_hash:
                return True
                
        return False
        
    def _check_if_file_exists_in_backend(self, filename: str, namespace: Optional[str] = None, hashed_identifier: Optional[str] = None) -> bool:
        """Check if file exists in backend by hashed identifier."""
        try:
            # Build namespace from user_id and project_name (same as upload/search)
            if namespace is None:
                user_id = self.get_or_create_device_id()
                project_name = self.get_project_name() or "default"
                user_id_safe = user_id.lower().replace(" ", "_")
                project_safe = project_name.lower().replace(" ", "_")
                namespace = f"{user_id_safe}-{project_safe}"
            
            if not hashed_identifier:
                return False

            count = self._get_backend_vector_count(hashed_identifier, namespace, filename=filename)
            if count is None:
                return False
            return count > 0
        except Exception as e:
            print(f"Error checking backend for file {filename}: {e}")
            return False  # If we can't check, assume it doesn't exist

    def _get_backend_vector_count(self, hashed_identifier: str, namespace: str, filename: Optional[str] = None) -> Optional[int]:
        """Return vector count for a file by hashed identifier, or None on failure."""
        if not hashed_identifier:
            return 0

        try:
            params = {"hashed_identifier": hashed_identifier, "namespace": namespace}
            response = requests.get(Config.CHECK_API_URL, params=params, timeout=20)
            if response.status_code == 200:
                result = response.json()
                count = result.get("vector_count")
                if count is None:
                    return None
                return int(count)

            name = filename or hashed_identifier[:8]
            print(f"Check endpoint failed for {name}: HTTP {response.status_code} {response.text}")
            return None
        except Exception as e:
            name = filename or hashed_identifier[:8]
            print(f"Error checking backend count for {name}: {e}")
            return None

    def _delete_backend_entry(self, filename: str, hashed_identifier: str, namespace: str):
        """Request backend deletion for a file's Pinecone data."""
        try:
            params = {
                "hashed_identifier": hashed_identifier,
                "filename": filename,
                "namespace": namespace
            }
            response = requests.delete(Config.DELETE_API_URL, params=params, timeout=15)
            if response.status_code != 200:
                print(f"Delete failed for {filename}: HTTP {response.status_code} {response.text}")
            else:
                print(f"Requested backend deletion for {filename}")
        except Exception as e:
            print(f"Error requesting deletion for {filename}: {e}")

    def _run_consistency_check(self, reason: str):
        """Sync local tracking with backend and remove dangling entries."""
        if not self.processed_files:
            return {"checked": 0, "removed": 0, "updated": 0}

        removed_count = 0
        checked_count = 0
        updated_count = 0
        now = time.time()

        for file_hash, info in list(self.processed_files.items()):
            filename = info.get("filename", "")
            filepath = info.get("filepath", "")
            namespace = info.get("namespace")
            expected_count = info.get("expected_vector_count")

            if not namespace:
                user_id = self.get_or_create_device_id()
                project_name = self.get_project_name() or "default"
                user_id_safe = user_id.lower().replace(" ", "_")
                project_safe = project_name.lower().replace(" ", "_")
                namespace = f"{user_id_safe}-{project_safe}"

            hashed_identifier = info.get("hashed_identifier") or self._get_hashed_identifier(filepath, namespace, filename)

            checked_count += 1

            if filepath and not os.path.exists(filepath):
                print(f"[Consistency] Missing local file: {filename}. Deleting from backend.")
                self._delete_backend_entry(filename, hashed_identifier, namespace)
                del self.processed_files[file_hash]
                removed_count += 1
                continue

            if filename:
                count = self._get_backend_vector_count(hashed_identifier, namespace, filename=filename)
                if count is None:
                    # Skip update if backend check failed
                    continue

                info['last_backend_check'] = now
                info['vector_count'] = count
                updated_count += 1

                if count <= 0:
                    print(f"[Consistency] Backend missing for: {filename}. Removing local record.")
                    del self.processed_files[file_hash]
                    removed_count += 1
                elif expected_count is not None and count != expected_count:
                    print(
                        f"[Consistency] Vector count mismatch for {filename}: "
                        f"expected {expected_count}, found {count}. Keeping local record."
                    )

        if removed_count > 0 or updated_count > 0:
            self._save_processed_files()
            self._update_file_status()

        if checked_count > 0:
            print(f"[Consistency] {reason}: checked {checked_count}, removed {removed_count}, updated {updated_count}")
        return {"checked": checked_count, "removed": removed_count, "updated": updated_count}

    def _verify_backend_records(self):
        """Manually verify backend vector counts for processed files."""
        if not self.processed_files:
            QMessageBox.information(self, "Verify Backend", "No processed files to verify.")
            return

        self.status_label.setText("Verifying backend records (forced)...")
        result = self._run_consistency_check("manual_verify")
        self._update_file_status()

        QMessageBox.information(
            self,
            "Verify Backend",
            f"Verification complete.\nChecked: {result.get('checked', 0)}\nUpdated: {result.get('updated', 0)}\nRemoved: {result.get('removed', 0)}"
        )
            
    def _process_upload_queue(self):
        """Process the next file in the upload queue."""
        if not self.upload_queue or self.is_uploading:
            return
            
        # Get next file from queue
        file_info = self.upload_queue.pop(0)
        self.current_upload = file_info
        self.is_uploading = True
        
        # Find and update the queued job to "processing" status
        temp_job_id = f"queued_{file_info['hash'][:8]}"
        if temp_job_id in self.current_jobs:
            self.current_jobs[temp_job_id]['status'] = 'processing'
            self._update_jobs_display()
        
        # Update UI
        remaining = len(self.upload_queue)
        filename = file_info['filename']
        self.status_label.setText(f"Uploading: {filename} ({remaining} remaining in queue)")
        
        # Build namespace from user_id and project_name
        user_id = self.get_or_create_device_id()
        project_name = self.get_project_name() or "default"
        
        # Simple namespace format: user_id-project_name (sanitized)
        user_id_safe = user_id.lower().replace(" ", "_")
        project_safe = project_name.lower().replace(" ", "_")
        namespace = f"{user_id_safe}-{project_safe}"
        
        # Upload the file
        self._upload_single_file(file_info, namespace)
            
    def _upload_single_file(self, file_info: Dict, namespace: str, retry_count: int = 0, max_retries: int = 3):
        """Upload a single file to the backend with retry logic."""
        filepath = file_info['filepath']
        filename = file_info['filename']
        file_hash = file_info['hash']
        
        try:
            if retry_count > 0:
                print(f"[Upload] Retry attempt {retry_count}/{max_retries} for {filename}")
            else:
                print(f"[Upload] Starting upload for {filename}")
            print(f"[Upload] File path: {filepath}")
            print(f"[Upload] Namespace: {namespace}")
            
            # Get file size first
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            print(f"[Upload] File size: {file_size_mb:.2f} MB")
                
            # Upload to backend - use file object directly for streaming
            # This is more memory efficient and may help with connection stability
            with open(filepath, 'rb') as f:
                files = {"file": (filename, f, "video/mp4")}
                data = {"namespace": namespace, "original_filepath": filepath}
                
                self.status_label.setText(f"Uploading {filename}... (attempt {retry_count + 1})")
                print(f"[Upload] Sending POST request to {Config.UPLOAD_API_URL}")
                
                # Use a session for better connection management
                session = requests.Session()
                # Increase timeout for larger files (calculate based on file size)
                # Allow at least 1 minute per MB, minimum 60 seconds, max 10 minutes
                upload_timeout = min(600, max(60, int(file_size_mb * 60)))
                print(f"[Upload] Using timeout: {upload_timeout} seconds")
                
                response = session.post(
                    Config.UPLOAD_API_URL, 
                    files=files, 
                    data=data, 
                    timeout=upload_timeout,
                    stream=False  # Don't stream response, we need the full response
                )
            
            print(f"[Upload] Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id")
                
                if job_id:
                    print(f"[Upload] ✅ Upload successful, job_id: {job_id}")
                    temp_job_id = f"queued_{file_hash[:8]}"
                    if temp_job_id in self.current_jobs:
                        del self.current_jobs[temp_job_id]
                    # Replace temp job with real job
                    job_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'file_hash': file_hash,
                        'status': 'processing',
                        'namespace': namespace
                    }
                    
                    self.current_jobs[job_id] = job_info
                    self.job_tracker.add_job(job_id, job_info)
                    self._update_jobs_display()
                    
                    self.status_label.setText(f"Upload started: {filename}")
                else:
                    error_msg = f"Upload failed for {filename}: No job ID returned. Response: {result}"
                    print(f"[Upload] ❌ {error_msg}")
                    QMessageBox.warning(self, "Error", error_msg)
                    # Continue with next upload even if this one failed
                    self._on_upload_completed(False)
            else:
                error_msg = f"Upload failed for {filename}: HTTP {response.status_code}\n{response.text}"
                print(f"[Upload] ❌ {error_msg}")
                QMessageBox.warning(self, "Error", error_msg)
                temp_job_id = f"queued_{file_hash[:8]}"
                if temp_job_id in self.current_jobs:
                    del self.current_jobs[temp_job_id]
                # Continue with next upload even if this one failed
                self._on_upload_completed(False)
                
        except (requests.exceptions.ConnectionError, requests.exceptions.ProtocolError) as e:
            # Retry connection errors with exponential backoff
            if retry_count < max_retries:
                wait_time = (2 ** retry_count) * 2  # 2, 4, 8 seconds
                error_msg = f"Connection error uploading {filename} (attempt {retry_count + 1}/{max_retries + 1}): {str(e)}"
                print(f"[Upload] ⚠️ {error_msg}")
                print(f"[Upload] Retrying in {wait_time} seconds...")
                self.status_label.setText(f"Connection error, retrying in {wait_time}s...")
                
                # Wait before retry
                time.sleep(wait_time)
                
                # Retry the upload
                self._upload_single_file(file_info, namespace, retry_count + 1, max_retries)
            else:
                error_msg = f"Connection error uploading {filename} after {max_retries + 1} attempts: {str(e)}"
                print(f"[Upload] ❌ {error_msg}")
                print(traceback.format_exc())
                temp_job_id = f"queued_{file_hash[:8]}"
                if temp_job_id in self.current_jobs:
                    del self.current_jobs[temp_job_id]
                QMessageBox.critical(self, "Network Error", f"{error_msg}\n\nPlease check your internet connection and try again.")
                self._on_upload_completed(False)
        except requests.exceptions.Timeout as e:
            # Retry timeout errors too
            if retry_count < max_retries:
                wait_time = (2 ** retry_count) * 2
                error_msg = f"Upload timeout for {filename} (attempt {retry_count + 1}/{max_retries + 1})"
                print(f"[Upload] ⚠️ {error_msg}")
                print(f"[Upload] Retrying in {wait_time} seconds...")
                self.status_label.setText(f"Timeout, retrying in {wait_time}s...")
                
                time.sleep(wait_time)
                
                self._upload_single_file(file_info, namespace, retry_count + 1, max_retries)
            else:
                error_msg = f"Upload timeout for {filename} after {max_retries + 1} attempts: {str(e)}"
                print(f"[Upload] ❌ {error_msg}")
                print(traceback.format_exc())
                temp_job_id = f"queued_{file_hash[:8]}"
                if temp_job_id in self.current_jobs:
                    del self.current_jobs[temp_job_id]
                QMessageBox.critical(self, "Upload Timeout", f"{error_msg}\n\nFile may be too large or connection too slow.")
                self._on_upload_completed(False)
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error uploading {filename}: {str(e)}"
            print(f"[Upload] ❌ {error_msg}")
            print(traceback.format_exc())
            temp_job_id = f"queued_{file_hash[:8]}"
            if temp_job_id in self.current_jobs:
                del self.current_jobs[temp_job_id]
            QMessageBox.critical(self, "Network Error", error_msg)
            self._on_upload_completed(False)
        except FileNotFoundError as e:
            error_msg = f"File not found: {filepath}\n{str(e)}"
            print(f"[Upload] ❌ {error_msg}")
            print(traceback.format_exc())
            temp_job_id = f"queued_{file_hash[:8]}"
            if temp_job_id in self.current_jobs:
                del self.current_jobs[temp_job_id]
            QMessageBox.critical(self, "File Error", error_msg)
            self._on_upload_completed(False)
        except Exception as e:
            error_msg = f"Failed to upload {filename}: {str(e)}"
            print(f"[Upload] ❌ {error_msg}")
            print(traceback.format_exc())
            temp_job_id = f"queued_{file_hash[:8]}"
            if temp_job_id in self.current_jobs:
                del self.current_jobs[temp_job_id]
            QMessageBox.critical(self, "Error", error_msg)
            self._on_upload_completed(False)
            
    def _on_job_completed(self, job_id: str, result: dict):
        """Handle job completion."""
        if job_id in self.current_jobs:
            job_info = self.current_jobs[job_id]
            filename = job_info['filename']
            filepath = job_info['filepath']
            file_hash = job_info['file_hash']
            namespace = job_info['namespace']
            hashed_identifier = self._get_hashed_identifier(filepath, namespace, filename)
            expected_vectors = None
            try:
                if isinstance(result, dict) and result.get("chunks") is not None:
                    expected_vectors = int(result.get("chunks"))
            except (TypeError, ValueError):
                expected_vectors = None
            
            # Mark file as processed (keep existing tracking)
            self.processed_files[file_hash] = {
                'filename': filename,
                'filepath': filepath,
                'job_id': job_id,
                'namespace': namespace,
                'hashed_identifier': hashed_identifier,
                'processed_at': time.time(),
                'result': result,
                'backend_miss_count': 0,
                'last_backend_check': None,
                'expected_vector_count': expected_vectors,
                'vector_count': expected_vectors
            }
            self._save_processed_files()
            
            # Remove from current jobs
            del self.current_jobs[job_id]
            self._update_jobs_display()
            self._update_file_status()
            
            self.status_label.setText(f"Completed: {filename}")
            print(f"✅ Job {job_id} completed successfully for {filename}")
            
            # Continue with next upload in queue
            self._on_upload_completed(True)
            # Skip immediate consistency check; backend counts can lag right after upload
            
    def _on_job_failed(self, job_id: str, error: str):
        """Handle job failure."""
        if job_id in self.current_jobs:
            job_info = self.current_jobs[job_id]
            filename = job_info['filename']
            file_hash = job_info['file_hash']
            
            print(f"❌ Job {job_id} failed for {filename}: {error}")
            
            # Before marking as failed, check if file actually exists in backend
            # (in case job tracking failed but processing succeeded)
            namespace = job_info.get('namespace')
            if self._check_if_file_exists_in_backend(
                filename,
                namespace=namespace,
                hashed_identifier=self._get_hashed_identifier(job_info.get('filepath', ''), namespace or "", filename)
            ):
                print(f"🔄 Job failed but file {filename} found in backend - marking as completed")
                
                # Mark as processed since it's actually in the backend
                self.processed_files[file_hash] = {
                    'filename': filename,
                    'filepath': job_info.get('filepath', ''),
                    'job_id': job_id,
                    'namespace': namespace,
                    'hashed_identifier': self._get_hashed_identifier(job_info.get('filepath', ''), namespace or "", filename),
                    'processed_at': time.time(),
                    'result': {'status': 'recovered_from_backend', 'error': error}
                }
                self._save_processed_files()
                
                # Remove from current jobs
                del self.current_jobs[job_id]
                self._update_jobs_display()
                self._update_file_status()
                
                self.status_label.setText(f"Recovered: {filename} (found in backend)")
                
                # Continue with next upload
                self._on_upload_completed(True)
                return
            
            # Actually failed - remove from current jobs
            del self.current_jobs[job_id]
            self._update_jobs_display()
            
            self.status_label.setText(f"Failed: {filename}")
            QMessageBox.warning(self, "Upload Failed", f"Processing failed for {filename}:\n{error}")
            
            # Continue with next upload in queue even if this one failed
            self._on_upload_completed(False)
            
    def _on_upload_completed(self, success: bool):
        """Handle completion of a single upload."""
        self.is_uploading = False
        self.current_upload = None
        
        # Process next file in queue
        if self.upload_queue:
            QTimer.singleShot(Config.QUEUE_DELAY, self._process_upload_queue)
        else:
            self.status_label.setText("All uploads completed!")
            self._update_file_status()
            
    def _update_jobs_display(self):
        """Update the jobs list display."""
        self.jobs_list.clear()
        
        for job_id, job_info in self.current_jobs.items():
            filename = job_info['filename']
            status = job_info.get('status', 'processing')
            
            # Add status emoji for better visual feedback
            if status == 'queued':
                status_display = "⏳ queued"
            elif status == 'processing':
                status_display = "🔄 processing"
            elif status == 'completed':
                status_display = "✅ completed"
            elif status == 'failed':
                status_display = "❌ failed"
            else:
                status_display = status
            
            item_text = f"{filename} - {status_display}"
            item = QListWidgetItem(item_text)
            self.jobs_list.addItem(item)
            
    def _perform_search(self):
        """Perform semantic search."""
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a search query")
            return
            
        # Build namespace from user_id and project_name (same as upload)
        user_id = self.get_or_create_device_id()
        project_name = self.get_project_name() or "default"
        user_id_safe = user_id.lower().replace(" ", "_")
        project_safe = project_name.lower().replace(" ", "_")
        namespace = f"{user_id_safe}-{project_safe}"
        
        self.status_label.setText(f"Searching for: {query}")
        self.btn_search.setEnabled(False)
        
        try:
            # Perform search using same approach as Streamlit
            params = {"query": query, "namespace": namespace}
            response = requests.get(Config.SEARCH_API_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                self._display_search_results(results, query)
                self.status_label.setText(f"Found {len(results)} results for: {query}")
            else:
                QMessageBox.warning(self, "Search Error", f"Search failed: {response.status_code}\n{response.text}")
                self.status_label.setText("Search failed")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed: {str(e)}")
            self.status_label.setText("Search error")
        finally:
            self.btn_search.setEnabled(True)
            
    def _display_search_results(self, results: List[Dict], query: str):
        """Display search results."""
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            child = self.results_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
                
        if not results:
            no_results = QLabel("No results found")
            no_results.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(no_results)
            return
            
        # Display results
        for i, result in enumerate(results[:10]):  # Limit to top 10
            result_widget = self._create_result_widget(result, i)
            self.results_layout.addWidget(result_widget)
            
        # Add stretch to push results to top
        self.results_layout.addStretch()
        
    def _create_result_widget(self, result: Dict, index: int) -> QWidget:
        """Create a widget for a single search result."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setMaximumHeight(120)
        
        layout = QHBoxLayout()
        
        # Result info
        info_layout = QVBoxLayout()
        
        metadata = result.get('metadata', {})
        filename = metadata.get('file_filename', 'Unknown')
        score = result.get('score', 0)
        start_time = metadata.get('start_time_s', 0)
        end_time = metadata.get('end_time_s', 0)
        
        # Title
        title_label = QLabel(f"<b>{filename}</b>")
        info_layout.addWidget(title_label)
        
        # Details
        details = f"Score: {score:.3f} | Time: {start_time:.1f}s - {end_time:.1f}s"
        details_label = QLabel(details)
        details_label.setStyleSheet("color: gray; font-size: 10px;")
        info_layout.addWidget(details_label)
        
        # Add to timeline button
        btn_add = QPushButton("Add to Timeline")
        btn_add.setMaximumWidth(120)
        btn_add.clicked.connect(lambda: self._add_result_to_timeline(result))
        
        layout.addLayout(info_layout)
        layout.addStretch()
        layout.addWidget(btn_add)
        
        widget.setLayout(layout)
        return widget
        
    def _add_result_to_timeline(self, result: Dict):
        """Add a search result to the timeline."""
        if not resolve:
            QMessageBox.warning(self, "Error", "Resolve API not available.")
            return
        
        # Refresh media pool to ensure we have latest clips
        self._refresh_media_pool(debug=True)
        print(f"[Timeline] clip_map entries: {len(self.clip_map)}")
            
        def _normalize_path(path_value: Optional[str]) -> Optional[str]:
            if not path_value:
                return None
            try:
                return os.path.normcase(os.path.normpath(path_value))
            except Exception:
                return path_value

        metadata = result.get('metadata', {})
        filename = metadata.get('file_filename', 'Unknown')
        file_path = metadata.get('file_path')
        normalized_file_path = _normalize_path(file_path)
        print(f"[Timeline] Target filename: {filename}")
        print(f"[Timeline] Target file_path: {file_path}")
        start_time = metadata.get('start_time_s', 0)
        end_time = metadata.get('end_time_s', 0)

        # Validate time range
        try:
            if float(end_time) <= float(start_time):
                QMessageBox.warning(self, "Error", f"Invalid time range for {filename}: {start_time} - {end_time}")
                return
        except Exception:
            QMessageBox.warning(self, "Error", f"Invalid time range metadata for {filename}.")
            return
        
        # Find the clip in media pool
        matching_clip = None
        matching_clip_info = None
        if normalized_file_path:
            for _, clip_info in self.clip_map.items():
                if isinstance(clip_info, list):
                    for entry in clip_info:
                        if _normalize_path(entry.get('filepath')) == normalized_file_path:
                            matching_clip_info = entry
                            matching_clip = entry.get('media_pool_item')
                            break
                else:
                    if _normalize_path(clip_info.get('filepath')) == normalized_file_path:
                        matching_clip_info = clip_info
                        matching_clip = clip_info.get('media_pool_item')
                if matching_clip:
                    break

        if not matching_clip:
            filename_lower = filename.lower()
            for clip_filename, clip_info in self.clip_map.items():
                clip_filename_lower = clip_filename.lower()
                if filename_lower in clip_filename_lower or clip_filename_lower in filename_lower:
                    matching_clip_info = clip_info
                    if isinstance(clip_info, list):
                        matching_clip = clip_info[0]['media_pool_item']
                    else:
                        matching_clip = clip_info['media_pool_item']
                    break
                
        if not matching_clip:
            # Debug logging to diagnose mismatches
            print("[Timeline] Failed to match clip in media pool")
            print(f"[Timeline] Result filename: {filename}")
            print(f"[Timeline] Result file_path: {file_path}")
            sample_paths = []
            total_paths = 0
            for _, clip_info in self.clip_map.items():
                if isinstance(clip_info, list):
                    for entry in clip_info:
                        path = entry.get('filepath')
                        if path:
                            total_paths += 1
                            if len(sample_paths) < 5:
                                sample_paths.append(path)
                else:
                    path = clip_info.get('filepath')
                    if path:
                        total_paths += 1
                        if len(sample_paths) < 5:
                            sample_paths.append(path)
            print(f"[Timeline] Media pool file paths: {total_paths} total")
            for p in sample_paths:
                print(f"[Timeline] Sample path: {p}")
            QMessageBox.warning(self, "Error", f"Could not find {filename} in media pool")
            return

        try:
            clip_name = matching_clip.GetName()
        except Exception:
            clip_name = None
        print(f"[Timeline] Matched clip: {clip_name or '<unnamed>'}")
            
        # Ensure timeline exists
        if not self._ensure_timeline():
            return
        try:
            timeline = project.GetCurrentTimeline()
            if timeline:
                project.SetCurrentTimeline(timeline)
            resolve.OpenPage("edit")
        except Exception:
            pass
            
        def _count_timeline_items(tl) -> Optional[int]:
            try:
                total = 0
                track_count = tl.GetTrackCount("video")
                for i in range(1, int(track_count) + 1):
                    items = tl.GetItemListInTrack("video", i) or []
                    total += len(items)
                return total
            except Exception:
                return None

        # Add to timeline
        try:
            # Ensure current folder is the root (some Resolve versions require it)
            try:
                root_folder = media_pool.GetRootFolder()
                if root_folder:
                    media_pool.SetCurrentFolder(root_folder)
            except Exception:
                pass

            # Determine clip length (frames) and fps
            clip_frames = None
            if matching_clip_info:
                for key in ("Frames", "Frame Count", "Duration"):
                    try:
                        val = matching_clip_info["media_pool_item"].GetClipProperty(key)
                        if val:
                            clip_frames = int(float(val))
                            break
                    except Exception:
                        pass
            if clip_frames is None:
                clip_frames = 1000

            fps = 24.0
            if isinstance(matching_clip_info, dict):
                fps = matching_clip_info.get("fps") or 24.0
            elif isinstance(matching_clip_info, list) and matching_clip_info:
                fps = matching_clip_info[0].get("fps") or 24.0

            start_frame = int(float(start_time) * float(fps))
            end_frame = int(float(end_time) * float(fps))
            start_frame = max(0, min(start_frame, clip_frames))
            end_frame = max(0, min(end_frame, clip_frames))
            if end_frame <= start_frame:
                end_frame = start_frame + 1

            print(f"[Timeline] Using fps={fps}, start_frame={start_frame}, end_frame={end_frame}, clip_frames={clip_frames}")

            # Ensure at least one video/audio track exists (AppendToTimeline can fail on empty timelines)
            try:
                if timeline and int(timeline.GetTrackCount("video") or 0) == 0:
                    timeline.AddTrack("video")
                    print("[Timeline] Added missing video track 1.")
                if timeline and int(timeline.GetTrackCount("audio") or 0) == 0:
                    timeline.AddTrack("audio")
                    print("[Timeline] Added missing audio track 1.")
            except Exception as e:
                print(f"[Timeline] Failed to ensure video track: {e}")

            # Best-effort track selection (blue source + red target) with diagnostics only.
            if timeline:
                enable_fn = getattr(timeline, "SetTrackEnable", None)
                autoselect_fn = getattr(timeline, "SetTrackAutoSelect", None)
                lock_fn = getattr(timeline, "SetTrackLock", None)
                print(
                    "[Timeline] Track methods:",
                    f"Enable={callable(enable_fn)}, AutoSelect={callable(autoselect_fn)}, Lock={callable(lock_fn)}"
                )
                for track_type in ("video", "audio"):
                    for name, fn, args in (
                        ("Enable", enable_fn, (track_type, 1, True)),
                        ("AutoSelect", autoselect_fn, (track_type, 1, True)),
                        ("Lock", lock_fn, (track_type, 1, False)),
                    ):
                        if not callable(fn):
                            continue
                        try:
                            result = fn(*args)
                            print(f"[Timeline] {name} {track_type}1 -> {result}")
                        except Exception as e:
                            print(f"[Timeline] {name} {track_type}1 failed: {e}")

            before_count = _count_timeline_items(timeline) if timeline else None
            print(f"[Timeline] Items before append: {before_count}")

            clip_info = {
                "mediaPoolItem": matching_clip,
                "startFrame": start_frame,
                "endFrame": end_frame,
            }
            result = media_pool.AppendToTimeline([clip_info])
            print(f"[Timeline] AppendToTimeline result: {result!r} (type={type(result).__name__})")

            def _is_append_success(value) -> bool:
                if value is True:
                    return True
                if isinstance(value, list):
                    return any(item is not None for item in value)
                return bool(value)

            success = _is_append_success(result)
            if not success:
                print("[Timeline] Timed append failed, trying full clip fallback.")
                result_fallback = media_pool.AppendToTimeline([matching_clip])
                print(f"[Timeline] Fallback AppendToTimeline result: {result_fallback!r} (type={type(result_fallback).__name__})")
                success = _is_append_success(result_fallback)

            after_count = _count_timeline_items(timeline) if timeline else None
            print(f"[Timeline] Items after append: {after_count}")

            if success:
                self.status_label.setText(f"Added {filename} ({start_time:.1f}s-{end_time:.1f}s) to timeline")
            else:
                print("[Timeline] AppendToTimeline returned False")
                QMessageBox.warning(
                    self,
                    "Failed to Add Clip",
                    "Resolve could not insert the clip. If this is an empty timeline, "
                    "please enable Edit -> Edit Options -> Automatically Create Tracks on Edit, "
                    "or drag any clip into the timeline once to initialize track patching."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add clip: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        if hasattr(self, 'job_tracker'):
            self.job_tracker.stop()
            self.job_tracker.wait()
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()
        event.accept()

    def _ensure_timeline(self):
        """Ensure a timeline exists, create one if needed."""
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

    def _build_clip_map(self, debug: bool = False):
        """Scan media pool clips that match the current filter and build a mapping.

        The filter used is the same as in `_action_append_chunk`: keep clips
        where `"Video"` appears in `GetClipProperty("Type")`.

        Returns a dict mapping `filename` -> { 'media_pool_item': <item>, 'fps': <float|None> }
        If multiple items share the same filename, the value becomes a list of such dicts.
        """
        if not resolve:
            return {}  # Return empty dict instead of raising error

        def _get_subfolders(folder):
            if not folder:
                return []
            for method in ("GetSubFolderList", "GetSubFolders"):
                try:
                    result = getattr(folder, method)()
                    if result is not None:
                        return result
                except Exception:
                    continue
            return []

        def _collect_clips(folder):
            if not folder:
                return []
            collected = []
            try:
                collected.extend(folder.GetClipList() or [])
            except Exception:
                pass
            for sub in _get_subfolders(folder):
                collected.extend(_collect_clips(sub))
            return collected

        root_folder = media_pool.GetRootFolder()
        clips = _collect_clips(root_folder)
        if debug:
            print(f"[MediaPool] Total clips found (pre-filter): {len(clips)}")

        # Apply same filter as the append action
        filtered = [c for c in clips if c and c.GetClipProperty("Type") and "Video" in c.GetClipProperty("Type")]
        if debug:
            print(f"[MediaPool] Video clips after filter: {len(filtered)}")

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

    def get_project_name(self) -> Optional[str]:
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
    
    def _show_processed_files(self):
        """Show processed files in a dialog window."""
        if not self.processed_files:
            QMessageBox.information(self, "Processed Files", "No files have been processed yet.")
            return
        
        try:
            # Create dialog window (modal to prevent it from closing)
            dialog = QDialog(self)
            dialog.setWindowTitle("Processed Files")
            dialog.resize(600, 400)
            dialog.setModal(True)  # Make it modal so it stays open
            
            layout = QVBoxLayout()
            
            # Header
            header = QLabel(f"<b>Processed Files ({len(self.processed_files)} total)</b>")
            layout.addWidget(header)
            
            # Scrollable list
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout()
            
            # Display each processed file
            for file_hash, file_info in sorted(self.processed_files.items(), 
                                              key=lambda x: x[1].get('processed_at', 0), 
                                              reverse=True):
                file_frame = QFrame()
                file_frame.setFrameStyle(QFrame.Shape.Box)
                file_layout = QVBoxLayout()
                
                filename = file_info.get('filename', 'Unknown')
                job_id = file_info.get('job_id', 'Unknown')
                processed_at = file_info.get('processed_at', 0)
                result = file_info.get('result', {})
                
                # Format timestamp
                if processed_at:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(processed_at)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = "Unknown"
                
                # File info
                file_label = QLabel(f"<b>{filename}</b>")
                file_layout.addWidget(file_label)
                
                info_label = QLabel(f"Job ID: {job_id} | Processed: {time_str}")
                info_label.setStyleSheet("color: gray; font-size: 9px;")
                file_layout.addWidget(info_label)
                
                # Status
                status = result.get('status', 'unknown')
                status_label = QLabel(f"Status: {status}")
                status_label.setStyleSheet("color: blue; font-size: 9px;")
                file_layout.addWidget(status_label)

                vector_count = file_info.get('vector_count')
                vector_text = f"Vectors: {vector_count}" if vector_count is not None else "Vectors: unknown"
                vector_label = QLabel(vector_text)
                vector_label.setStyleSheet("color: gray; font-size: 9px;")
                file_layout.addWidget(vector_label)
                
                file_frame.setLayout(file_layout)
                scroll_layout.addWidget(file_frame)
            
            scroll_layout.addStretch()
            scroll_widget.setLayout(scroll_layout)
            scroll.setWidget(scroll_widget)
            layout.addWidget(scroll)
            
            # Close button
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.accept)
            layout.addWidget(btn_close)
            
            dialog.setLayout(layout)
            dialog.exec()  # Use exec() instead of show() to make it modal
        except Exception as e:
            error_msg = f"Error showing processed files dialog: {e}\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to show processed files:\n{str(e)}")
    
    def _clear_processed_files(self):
        """Clear all processed files tracking."""
        if not self.processed_files:
            QMessageBox.information(self, "Clear Processed Files", "No processed files to clear.")
            return

        reply = QMessageBox.question(
            self,
            "Clear Processed Files",
            f"This will clear tracking for {len(self.processed_files)} processed files.\n\n"
            "Do you also want to delete their vectors from Pinecone?\n\n"
            "Yes = delete Pinecone + local\nNo = delete Pinecone only (keep local)\nCancel = do nothing",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Cancel:
            return

        if reply == QMessageBox.StandardButton.Yes:
            # Delete from backend before clearing local
            for _, info in list(self.processed_files.items()):
                filename = info.get("filename", "")
                filepath = info.get("filepath", "")
                namespace = info.get("namespace")
                if not namespace:
                    user_id = self.get_or_create_device_id()
                    project_name = self.get_project_name() or "default"
                    user_id_safe = user_id.lower().replace(" ", "_")
                    project_safe = project_name.lower().replace(" ", "_")
                    namespace = f"{user_id_safe}-{project_safe}"

                hashed_identifier = info.get("hashed_identifier") or self._get_hashed_identifier(filepath, namespace, filename)
                if filename and hashed_identifier:
                    self._delete_backend_entry(filename, hashed_identifier, namespace)

            self.processed_files.clear()
            self._save_processed_files()
            self._update_file_status()
            QMessageBox.information(self, "Cleared", "Processed files tracking has been cleared.")
            print("Processed files tracking cleared")
        elif reply == QMessageBox.StandardButton.No:
            # Delete from backend only, keep local so verification can prune
            for _, info in list(self.processed_files.items()):
                filename = info.get("filename", "")
                filepath = info.get("filepath", "")
                namespace = info.get("namespace")
                if not namespace:
                    user_id = self.get_or_create_device_id()
                    project_name = self.get_project_name() or "default"
                    user_id_safe = user_id.lower().replace(" ", "_")
                    project_safe = project_name.lower().replace(" ", "_")
                    namespace = f"{user_id_safe}-{project_safe}"

                hashed_identifier = info.get("hashed_identifier") or self._get_hashed_identifier(filepath, namespace, filename)
                if filename and hashed_identifier:
                    self._delete_backend_entry(filename, hashed_identifier, namespace)

            QMessageBox.information(
                self,
                "Deleted from Pinecone",
                "Pinecone data deleted. Local tracking kept for verification."
            )
            print("Pinecone data deleted; local tracking kept")

# --- Main Execution ---
if __name__ == "__main__":
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