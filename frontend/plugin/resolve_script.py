import sys

# Add Resolve's scripting module to Python path
if sys.platform == "darwin":  # macOS
    resolve_script_path = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
    sys.path.append(resolve_script_path)
elif sys.platform == "win32":  # Windows
    resolve_script_path = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules"
    sys.path.append(resolve_script_path)

try:
    import DaVinciResolveScript as dvr_script
    
    # Try to connect
    resolve = dvr_script.scriptapp("Resolve")
    
    if resolve:
        print("✓ Successfully connected to DaVinci Resolve!")
        pm = resolve.GetProjectManager()
        project = pm.GetCurrentProject()
        
        if project:
            print(f"✓ Current project: {project.GetName()}")
        else:
            print("⚠ No project currently open")
    else:
        print("✗ Could not connect to Resolve")
        
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure DaVinci Resolve is running")
    print("2. Make sure you have a project open")
    print("3. Check if the scripting modules path is correct")