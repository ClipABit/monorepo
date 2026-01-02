# ClipABit - DaVinci Resolve Plugin

A semantic video search plugin for DaVinci Resolve that allows you to search through your media pool using natural language queries and automatically add matching clips to your timeline.

## Features

### Core Functionality
- **Semantic Video Search**: Search your media pool using natural language (e.g., "woman walking", "car driving")
- **Smart Upload Management**: Track which files have been processed and upload only new files
- **Background Job Tracking**: Monitor upload and processing jobs with real-time status updates
- **Timeline Integration**: Add search results directly to your timeline with precise timing
- **Persistent Storage**: Local tracking of processed files across sessions

### User Interface
- **Split Panel Design**: Upload controls on the left, search and results on the right
- **File Status Display**: Shows total files, processed count, and new files available
- **Active Jobs Monitor**: Real-time display of upload and processing jobs
- **Search Results**: Visual display of top matching clips with scores and timing
- **One-Click Timeline Addition**: Add any search result to timeline with a single click

## Prerequisites

* **DaVinci Resolve** installed in default location
* **Python 3.12+** 
* **Required Python Packages**:
  * `PyQt6` - UI framework
  * `requests` - API communication
  * `watchdog` - File watching for development

## Setup & Installation

### 1. Install Dependencies
```bash
# Navigate to plugin directory
cd frontend/plugin

# Install via uv (recommended)
uv pip install

# Or install globally via pip
pip install PyQt6 requests watchdog
```

### 2. Development Setup (Hot Reload)
```bash
# Navigate to development utilities
cd utils/plugins/davinci

# Start file watcher for automatic updates
python watch_clipabit.py
```

This starts a file watcher that automatically copies the plugin to Resolve's Scripts directory when you make changes, enabling hot-reload development.

### 3. Running the Plugin
1. Open DaVinci Resolve
2. Open or create a project
3. Navigate to: `Workspace > Scripts > ClipABit`
4. The plugin window will open

## Usage

### Initial Setup
1. **Load Media**: Add video files to your DaVinci Resolve media pool
2. **Upload Files**: Click "Upload New Files" to process and index your media
3. **Monitor Jobs**: Watch the "Active Jobs" panel for processing status

### Searching & Timeline Integration
1. **Search**: Enter natural language queries in the search box
   - Examples: "person walking", "car on road", "sunset scene"
2. **Review Results**: Browse search results with similarity scores and timing
3. **Add to Timeline**: Click "Add to Timeline" on any result to insert the clip

### File Management
- **Upload New Files**: Only uploads files that haven't been processed
- **Upload All Files**: Re-uploads all files in media pool
- **Automatic Tracking**: Plugin remembers which files have been processed
- **Status Display**: Shows file counts and processing status

## Configuration

### Environment Settings
Set the `CLIPABIT_ENVIRONMENT` environment variable to control which backend to use:
- `dev` (default) - Development environment
- `prod` - Production environment  
- `staging` - Staging environment

### API Endpoints
The plugin automatically configures API endpoints based on the environment:
- Search: `https://clipabit01--{env}-server-search{suffix}.modal.run`
- Upload: `https://clipabit01--{env}-server-upload{suffix}.modal.run`
- Status: `https://clipabit01--{env}-server-status{suffix}.modal.run`

### Data Storage
- **Device ID**: Stored in platform-appropriate location for user identification
- **Processed Files**: JSON file tracking which media has been uploaded
- **Namespacing**: Each device + project combination gets its own namespace

## Architecture

### Backend Integration
- **Modal.com Deployment**: Serverless backend for video processing
- **Pinecone Vector Database**: Stores video embeddings for semantic search
- **R2 Storage**: Cloudflare R2 for video file storage
- **CLIP Embeddings**: OpenAI CLIP model for semantic understanding

### Local Components
- **PyQt6 Interface**: Modern, responsive UI
- **Background Job Tracking**: Non-blocking upload and status monitoring
- **File System Integration**: Direct access to Resolve's media pool
- **Persistent State**: Local storage for processed file tracking

### DaVinci Resolve Integration
- **Media Pool Access**: Reads video files from current project
- **Timeline Manipulation**: Adds clips with precise frame timing
- **Project Awareness**: Separate namespaces per project
- **Real-time Updates**: Monitors media pool changes

## Development

### File Structure
```
frontend/plugin/
├── clipabit.py          # Main plugin file
├── pyproject.toml       # Dependencies
├── README.md           # This file
└── uv.lock            # Dependency lock file

utils/plugins/davinci/
└── watch_clipabit.py   # Development file watcher
```

### Development Workflow
1. Make changes to `frontend/plugin/clipabit.py`
2. File watcher automatically copies to Resolve Scripts directory
3. Restart plugin in Resolve to see changes
4. Use Resolve Console (`Workspace > Console`) for debugging

### Key Classes
- **ClipABitApp**: Main application window and logic
- **JobTracker**: Background thread for monitoring upload jobs
- **Config**: Environment-based configuration management

## Troubleshooting

### Common Issues
- **"Resolve API not available"**: Plugin is running outside of DaVinci Resolve
- **Upload failures**: Check internet connection and API endpoints
- **No search results**: Ensure files have been uploaded and processed
- **Timeline errors**: Verify timeline exists or plugin will create one

### Debug Information
- Check Resolve Console for detailed logs
- Plugin status bar shows current operations
- Job tracker displays upload progress and errors

### File Locations
- **Windows**: `%APPDATA%\ClipABit\`
- **macOS**: `~/Library/Application Support/ClipABit/`
- **Linux**: `~/.config/clipabit/`

## Future Enhancements

- **Preview Thumbnails**: Visual previews of search results
- **Batch Operations**: Multi-select and batch timeline additions
- **Advanced Filters**: Filter by duration, resolution, or other metadata
- **Custom Namespaces**: User-defined organization schemes
- **Offline Mode**: Local-only search capabilities