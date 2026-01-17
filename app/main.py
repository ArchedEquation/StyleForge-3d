from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import subprocess
import uuid
import logging
from app.pipeline import process_3d_pipeline
from app.config import get_blender_exec
from contextlib import asynccontextmanager

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
STYLES_DIR = os.path.join(BASE_DIR, "styles")

# Ensure Directories
os.makedirs(TEMP_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StyleForgeAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup check
    # Blender removed. Python dependencies managed via pip.
    logger.info("StyleForge 3D (Python-native mode) initialized.")
    yield

app = FastAPI(title="StyleForge 3D", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STYLES_DIR), name="styles")

@app.get("/")
def read_root():
    return {"message": "Welcome to StyleForge 3D. Use /docs for API."}

@app.get("/styles")
def list_styles():
    """List available styles with IDs."""
    styles = []
    if os.path.exists(STYLES_DIR):
        for f in sorted(os.listdir(STYLES_DIR)):
            if f.startswith("style-") and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    sid = f.split("-")[1].split(".")[0]
                    styles.append({
                        "id": sid,
                        "filename": f,
                        "url": f"/static/{f}"
                    })
                except:
                    pass
    return {"styles": styles}

@app.post("/upload")
async def process_file(
    file: UploadFile = File(...),
    style_id: str = Form(...)
):
    """
    Upload a 3D file (OBJ/GLTF), process it, and apply style.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".obj", ".gltf", ".glb", ".fbx"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, f"input{ext}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Call the shared pipeline logic
        process_3d_pipeline(input_path, style_id, job_dir)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"job_id": job_id, "status": "completed", "download_url": f"/download/{job_id}"})

@app.get("/download/{job_id}")
def download_result(job_id: str):
    job_dir = os.path.join(TEMP_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
        
    final_glb = os.path.join(job_dir, "styleforge_result.glb")
    if os.path.exists(final_glb):
        return FileResponse(final_glb, filename=f"styleforge_{job_id}.glb")

    output_zip = os.path.join(job_dir, "result.zip")
    if not os.path.exists(output_zip):
        shutil.make_archive(os.path.join(job_dir, "result"), 'zip', job_dir)
    
    return FileResponse(output_zip, filename=f"result_{job_id}.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
