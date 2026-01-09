import sys
import os
import requests
import uuid
import hashlib
import platform
import json
import time
import io
from pathlib import Path
from typing import Dict, List, Optional

# Try to import PyQt6
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                 QLabel, QPushButton, QMessageBox, QLineEdit,
                                 QScrollArea, QFrame, QSplitter, QListWidget, QListWidgetItem)
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
    LIST_VIDEOS_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-list-videos{url_portion}.modal.run"
    
    # Namespace - use default namespace for consistency with backend
    NAMESPACE = "__default__"
    
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
                    print(f"Error checking job {job_id}: {e}")
                    
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
    resolve = app.GetResolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    media_pool = project.GetMediaPool()
    project_timeline = project.GetCurrentTimeline()
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
        self.search_results = []
        
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
            self.clip_map = self._build_clip_map()
            print(f"Found {len(self.clip_map)} clips in media pool")
        else:
            self.clip_map = {}
            print("Running without Resolve API - clip map disabled")
        
        # Setup UI
        self.setWindowTitle("ClipABit Plugin (Resolve 20)")
        self.resize(800, 600)
        self.init_ui()
        
        # Setup refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_media_pool)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # Setup job recovery timer to handle Modal container scaling issues
        self.job_recovery_timer = QTimer()
        self.job_recovery_timer.timeout.connect(self._recover_lost_jobs)
        self.job_recovery_timer.start(30000)  # Check every 30 seconds
        
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
        
        # Upload buttons
        self.btn_upload_new = QPushButton("Process New Files")
        self.btn_upload_new.setMinimumHeight(35)
        self.btn_upload_new.clicked.connect(lambda: self._upload_files(new_only=True))
        upload_layout.addWidget(self.btn_upload_new)
        
        self.btn_upload_all = QPushButton("Process All Files")
        self.btn_upload_all.setMinimumHeight(35)
        self.btn_upload_all.clicked.connect(lambda: self._upload_files(new_only=False))
        upload_layout.addWidget(self.btn_upload_all)
        
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
        device_path = self._get_device_id_path()
        return device_path.parent / "processed_files.json"
        
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
            
    def _refresh_media_pool(self):
        """Refresh media pool and update file status."""
        if not resolve:
            return
            
        self.clip_map = self._build_clip_map()
        self._update_file_status()
        
    def _recover_lost_jobs(self):
        """Recover jobs that may have completed but were lost due to Modal container scaling."""
        if not self.current_jobs:
            return  # No jobs to recover
            
        print(f"🔄 Checking {len(self.current_jobs)} active jobs for recovery...")
        
        jobs_to_recover = []
        for job_id, job_info in self.current_jobs.items():
            # Skip temporary queue jobs
            if job_info.get('temp_job'):
                continue
                
            filename = job_info['filename']
            
            # Check if this file now exists in backend (meaning job completed but we lost tracking)
            if self._check_if_file_exists_in_backend(filename):
                jobs_to_recover.append((job_id, job_info))
                
        # Recover found jobs
        for job_id, job_info in jobs_to_recover:
            filename = job_info['filename']
            file_hash = job_info['file_hash']
            
            print(f"🔄 Recovering lost job: {filename}")
            
            # Mark as processed
            self.processed_files[file_hash] = {
                'filename': filename,
                'job_id': job_id,
                'processed_at': time.time(),
                'result': {'status': 'recovered_lost_job', 'recovered_at': time.time()}
            }
            self._save_processed_files()
            
            # Remove from current jobs
            del self.current_jobs[job_id]
            
        if jobs_to_recover:
            self._update_jobs_display()
            self._update_file_status()
            self.status_label.setText(f"Recovered {len(jobs_to_recover)} lost jobs")
            print(f"✅ Recovered {len(jobs_to_recover)} lost jobs")
        
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
        
        # Update button states - disable during upload
        self.btn_upload_new.setEnabled(new_count > 0 and not self.is_uploading)
        self.btn_upload_all.setEnabled(total_files > 0 and not self.is_uploading)
        self.btn_clear_queue.setEnabled(queued_count > 0 or self.is_uploading)
        
    def _upload_files(self, new_only: bool = True):
        """Upload files to backend using queue system."""
        if not resolve:
            QMessageBox.warning(self, "Error", "Resolve API not available.")
            return
            
        files_to_upload = []
        
        for filename, clip_info in self.clip_map.items():
            clips = clip_info if isinstance(clip_info, list) else [clip_info]
            
            for clip in clips:
                filepath = clip.get('filepath')
                if not filepath or not os.path.exists(filepath):
                    continue
                    
                file_hash = self._get_file_hash(filepath)
                
                if new_only and file_hash in self.processed_files:
                    continue  # Skip already processed files
                
                # ALSO check if file is currently being processed
                if self._is_file_being_processed(file_hash):
                    print(f"Skipping {filename} - already in processing queue")
                    continue
                
                # ALSO check if file already exists in backend (in case local tracking failed)
                if new_only and self._check_if_file_exists_in_backend(filename):
                    print(f"Skipping {filename} - already exists in backend")
                    # Mark it as processed locally so we don't check again
                    self.processed_files[file_hash] = {
                        'filename': filename,
                        'job_id': 'backend_existing',
                        'processed_at': time.time(),
                        'result': {'status': 'found_in_backend'}
                    }
                    self._save_processed_files()
                    continue
                    
                files_to_upload.append({
                    'filepath': filepath,
                    'filename': filename,
                    'hash': file_hash,
                    'clip_info': clip
                })
        
        if not files_to_upload:
            msg = "No new files to upload" if new_only else "No files to upload"
            QMessageBox.information(self, "Info", msg)
            return
            
        # Add files to upload queue
        self.upload_queue.extend(files_to_upload)
        
        # IMMEDIATELY add all files to jobs display with "queued" status
        for file_info in files_to_upload:
            # Create a temporary job ID for display purposes
            temp_job_id = f"queued_{file_info['hash'][:8]}"
            self.current_jobs[temp_job_id] = {
                'filename': file_info['filename'],
                'filepath': file_info['filepath'],
                'file_hash': file_info['hash'],
                'status': 'queued',
                'temp_job': True  # Mark as temporary for queue display
            }
        
        # Update jobs display immediately
        self._update_jobs_display()
        
        # Update status
        total_queued = len(self.upload_queue)
        self.status_label.setText(f"Added {len(files_to_upload)} files to upload queue ({total_queued} total)")
        
        # Start processing queue if not already uploading
        if not self.is_uploading:
            self._process_upload_queue()
            
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
        
    def _check_if_file_exists_in_backend(self, filename: str) -> bool:
        """Check if file already exists in backend by searching for it."""
        try:
            # Search for the exact filename in the backend with longer timeout for cold starts
            params = {"query": filename, "namespace": Config.NAMESPACE}
            response = requests.get(Config.SEARCH_API_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                
                # Check if any result matches this exact filename
                for result in results:
                    metadata = result.get('metadata', {})
                    backend_filename = metadata.get('file_filename', '')
                    if backend_filename == filename:
                        print(f"File {filename} already exists in backend (found {len(results)} total results)")
                        return True
                        
            return False
        except requests.exceptions.Timeout:
            print(f"Timeout checking backend for file {filename} - assuming it doesn't exist")
            return False  # If timeout, assume it doesn't exist to avoid blocking uploads
        except Exception as e:
            print(f"Error checking backend for file {filename}: {e}")
            return False  # If we can't check, assume it doesn't exist
            
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
        
        # Use configured namespace
        namespace = Config.NAMESPACE
        
        # Upload the file
        self._upload_single_file(file_info, namespace)
            
    def _upload_single_file(self, file_info: Dict, namespace: str):
        """Upload a single file to the backend."""
        filepath = file_info['filepath']
        filename = file_info['filename']
        file_hash = file_info['hash']
        
        # Remove the temporary "queued" job entry
        temp_job_id = f"queued_{file_hash[:8]}"
        if temp_job_id in self.current_jobs:
            del self.current_jobs[temp_job_id]
        
        try:
            # Read file
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
                
            # Upload to backend
            files = {"file": (filename, io.BytesIO(file_bytes), "video/mp4")}
            data = {"namespace": namespace}
            
            self.status_label.setText(f"Uploading {filename}...")
            
            response = requests.post(Config.UPLOAD_API_URL, files=files, data=data, timeout=Config.UPLOAD_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id")
                
                if job_id:
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
                    QMessageBox.warning(self, "Error", f"Upload failed for {filename}: No job ID returned")
                    # Continue with next upload even if this one failed
                    self._on_upload_completed(False)
            else:
                QMessageBox.warning(self, "Error", f"Upload failed for {filename}: {response.status_code}")
                # Continue with next upload even if this one failed
                self._on_upload_completed(False)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to upload {filename}: {str(e)}")
            self._on_upload_completed(False)
            
    def _on_job_completed(self, job_id: str, result: dict):
        """Handle job completion."""
        if job_id in self.current_jobs:
            job_info = self.current_jobs[job_id]
            filename = job_info['filename']
            file_hash = job_info['file_hash']
            
            # Mark file as processed
            self.processed_files[file_hash] = {
                'filename': filename,
                'job_id': job_id,
                'processed_at': time.time(),
                'result': result
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
            
    def _on_job_failed(self, job_id: str, error: str):
        """Handle job failure."""
        if job_id in self.current_jobs:
            job_info = self.current_jobs[job_id]
            filename = job_info['filename']
            file_hash = job_info['file_hash']
            
            print(f"❌ Job {job_id} failed for {filename}: {error}")
            
            # Before marking as failed, check if file actually exists in backend
            # (in case job tracking failed but processing succeeded)
            if self._check_if_file_exists_in_backend(filename):
                print(f"🔄 Job failed but file {filename} found in backend - marking as completed")
                
                # Mark as processed since it's actually in the backend
                self.processed_files[file_hash] = {
                    'filename': filename,
                    'job_id': job_id,
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
            
        # Use configured namespace
        namespace = Config.NAMESPACE
        
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
            
        metadata = result.get('metadata', {})
        filename = metadata.get('file_filename', 'Unknown')
        start_time = metadata.get('start_time_s', 0)
        end_time = metadata.get('end_time_s', 0)
        
        # Find the clip in media pool
        matching_clip = None
        for clip_filename, clip_info in self.clip_map.items():
            if filename in clip_filename or clip_filename in filename:
                if isinstance(clip_info, list):
                    matching_clip = clip_info[0]['media_pool_item']
                else:
                    matching_clip = clip_info['media_pool_item']
                break
                
        if not matching_clip:
            QMessageBox.warning(self, "Error", f"Could not find {filename} in media pool")
            return
            
        # Ensure timeline exists
        if not self._ensure_timeline():
            return
            
        # Convert seconds to frames (assuming 24fps, should get from clip)
        fps = 24  # Default fallback
        if isinstance(clip_info, dict):
            fps = clip_info.get('fps', 24) or 24
        elif isinstance(clip_info, list) and clip_info:
            fps = clip_info[0].get('fps', 24) or 24
            
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Add to timeline
        try:
            result = media_pool.AppendToTimeline([{
                "mediaPoolItem": matching_clip,
                "startFrame": start_frame,
                "endFrame": end_frame
            }])
            
            if result:
                self.status_label.setText(f"Added {filename} ({start_time:.1f}s-{end_time:.1f}s) to timeline")
            else:
                QMessageBox.warning(self, "Error", "Failed to add clip to timeline")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add clip: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        if hasattr(self, 'job_tracker'):
            self.job_tracker.stop()
            self.job_tracker.wait()
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()
        if hasattr(self, 'job_recovery_timer'):
            self.job_recovery_timer.stop()
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

    def _build_clip_map(self):
        """Scan media pool clips that match the current filter and build a mapping.

        The filter used is the same as in `_action_append_chunk`: keep clips
        where `"Video"` appears in `GetClipProperty("Type")`.

        Returns a dict mapping `filename` -> { 'media_pool_item': <item>, 'fps': <float|None> }
        If multiple items share the same filename, the value becomes a list of such dicts.
        """
        if not resolve:
            return {}  # Return empty dict instead of raising error

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