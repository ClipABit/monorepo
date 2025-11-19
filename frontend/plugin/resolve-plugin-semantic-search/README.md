Semantic Search Resolve Plugin
==============================

This is a DaVinci Resolve Workflow Integration plugin that adds a simple UI for entering a semantic search query. It packages into a `.drfx` for easy installation. It will display the timestamps of the most similar searches and then be able to help locate the desired query.

Structure
---------

- `plugin/manifest.json`: Plugin manifest.
- `plugin/Scripts/semantic_search.py`: Entry script (Python 3) using Resolve's Fusion UI Manager.
- `plugin/UI/semantic_search.ui`: Optional Qt `.ui` placeholder (not required by the script).
- `build_drfx.sh`: Builds `resolve-plugin-semantic-search.drfx`.

Build
-----

1) Make the build script executable:

```bash
chmod +x ./build_drfx.sh
```

2) Build the package:

```bash
./build_drfx.sh
```

This generates:

```
./resolve-plugin-semantic-search.drfx
```

Install
-------

- For Workflow Integration (recommended for this plugin), use the installer script on macOS:

```bash
chmod +x ./install_local.sh
./install_local.sh
```

- To install as a Script (appears under Workspace → Scripts → <Page>):

```bash
./install_local.sh --script --page Edit
# Pages: Comp | Edit | Color | Deliver | Media | Cut | Fairlight
```

- To uninstall:

```bash
# Uninstall Workflow Integration
./install_local.sh --uninstall
# Uninstall Script install
./install_local.sh --script --uninstall
```

Run
---

After installing, restart Resolve. For Workflow Integration, look for the entry on the Media page (Resolve decides exact placement based on the manifest). For Script install, use: Workspace → Scripts → <Page> → semantic_search.py (use the page you chose with `--page`, e.g., Edit). The current script opens a minimal dialog where you can enter a query and prints it to the console.

Notes
-----

- Manifest json file is a commonly used schema for Resolve workflow integrations. This is how it knows that our plug in isn't a colour grading specific one or something else that isn't relevant for our use-cases.
- Next steps --> integrate backend or embeddings index inside `semantic_search.py`.


