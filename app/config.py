import os
import shutil
import sys

# Try to find blender in PATH
BLENDER_EXEC = shutil.which("blender")

# If not in PATH, try common locations (add more if needed)
if not BLENDER_EXEC:
    common_paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.0\blender.exe",
    ]
    for p in common_paths:
        if os.path.exists(p):
            BLENDER_EXEC = p
            break

# MANUAL OVERRIDE: If still not found, set it here explicitly
# BLENDER_EXEC = r"C:\Path\To\blender.exe"

if not BLENDER_EXEC:
    # Fallback to just "blender" and hope for the best, or None
    BLENDER_EXEC = "blender"

def get_blender_exec():
    if not BLENDER_EXEC or (BLENDER_EXEC != "blender" and not os.path.exists(BLENDER_EXEC)):
        # If it's the default string "blender" we can't easily check existence without which, 
        # but if we are here, which() failed.
        pass
    return BLENDER_EXEC
