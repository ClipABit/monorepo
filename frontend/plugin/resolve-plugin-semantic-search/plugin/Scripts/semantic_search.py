#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic Search Workflow Integration entry point for DaVinci Resolve.
This script registers a simple UI using Fusion's UI Manager to collect a query.
"""

from __future__ import annotations

import sys
import os
from typing import Optional


def get_resolve_app():
    """
    Obtain the Resolve application handle in a version-tolerant way.
    """
    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except ImportError:
        # Fallback path: Resolve typically places this near the script location
        script_paths = [
            os.path.join(os.path.expanduser("~"),
                         "Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts"),
            "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts",
            os.path.dirname(os.path.abspath(__file__)),
        ]
        for candidate in script_paths:
            candidate = os.path.abspath(candidate)
            if candidate not in sys.path:
                sys.path.append(candidate)
        try:
            import DaVinciResolveScript as dvr  # type: ignore
        except ImportError:
            dvr = None
    if dvr is None:
        return None
    return dvr.scriptapp("Resolve")


def show_ui(resolve_app) -> None:
    """
    Build a minimal UI to capture a search query and print it to the console.
    """
    fusion = resolve_app.Fusion()
    ui_mgr = fusion.UIManager
    dispatcher = bmd.UIDispatcher(ui_mgr)  # type: ignore  # Provided by Resolve runtime

    window_title = "Semantic Search"
    width, height = 520, 120

    main_window = ui_mgr.CreateWindow({
        "ID": "SemanticSearchWindow",
        "WindowTitle": window_title,
        "Geometry": [100, 100, width, height],
    },
        ui_mgr.VGroup({
            "Spacing": 8,
            "Weight": 1,
        }, [
            ui_mgr.Label({"Text": "Enter semantic query:"}),
            ui_mgr.LineEdit({"ID": "QueryInput", "PlaceholderText": "e.g., find all close-up shots of a person smiling"}),
            ui_mgr.HGroup({"Weight": 0, "Spacing": 8}, [
                ui_mgr.Button({"ID": "RunSearch", "Text": "Search"}),
                ui_mgr.Button({"ID": "Close", "Text": "Close"}),
            ]),
        ])
    )

    itm = main_window.GetItems()

    def on_close(ev):
        dispatcher.ExitLoop()

    def on_search(ev):
        query_text = itm["QueryInput"].Text or ""
        print("[SemanticSearch] query:", query_text)
        # Placeholder action: demonstrate basic access to the current project and bins
        project_manager = resolve_app.GetProjectManager()
        project = project_manager.GetCurrentProject() if project_manager else None
        project_name = project.GetName() if project else "(no project)"
        print(f"[SemanticSearch] active project: {project_name}")
        # Extend here: integrate with your backend or embeddings index

    main_window.On["Close"] = on_close
    itm["Close"].Pressed = on_close
    itm["RunSearch"].Clicked = on_search

    main_window.Show()
    dispatcher.RunLoop()
    main_window.Hide()


def main() -> None:
    resolve = get_resolve_app()
    if resolve is None:
        print("Could not load DaVinci Resolve scripting environment.")
        sys.exit(1)
    show_ui(resolve)


if __name__ == "__main__":
    main()


