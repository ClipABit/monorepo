# ClipAbit - DaVinci Resolve Script Development Workflow

This document outlines the setup and development workflow for the **ClipAbit** DaVinci Resolve script, utilizing a file watcher for a continuous integration/hot-reload development experience.

## Prerequisites

* **DaVinci Resolve** (DR) installed. Follow the instructions [on their website](https://www.blackmagicdesign.com/products/davinciresolve) ensure you download it in the default location for everything else to work
* **Python 3.12+** installed.
* **Python Packages**:
  * `watchdog` (Required for the file watcher).

### Setup 
First install all the necessary packages by running:
```bash
# Install via pip (or uv pip)
uv pip install
```

And make sure to install all the ui and requests library globally

```bash
pip install PyQt6
```

Then open a terminal in the utils directory for the plugin by running the below commands in the root directory of the project.

```bash
cd /utils/plugins/davinci
python3 ./watch_clipabit.py
```

This initializes a script in the directory that resolve looks at to run scripts from its native console.

Further as you *save* updates to the script code, the ```watch_clipabit.py``` process will automatically update the copy of the code in the resolve scripts directory. This allows for you to update and commit everything from within the repo.

To run the "plugin" in resolve you need to first open a project. Then click:
```Workspace > Scripts > ClipABit```

This opens a tkinter application that is the ClipABit plugin for resolve

### Notes for Development
* Make sure you close and rerun the script every time that you make updates to the plugin
* Open the resolve console at: ```Workspace > console``` for logs


### Currently Working On . . .
* Writing a process media pool function that embeds the entire media pool of your project and uploads it to the pinecone database.
  * Each user that uses the plugin will be allocated an index that identifies their device and a namespace for each project within that index.
* Implementing search so that the user can search for the most appropriate video within the media pool and click an append to timeline button to instantly add the specific seen that matches.
  * Need to ensure that the plugin displays previews of each search result shown.
* Would like to test out qtpy for ui because that is what resolve works with and thus we can avoid weird errors that way.
* create an installer so that all the dependencies required for resolve to run the script is initialized