# StyleForge 3D Backend

This project creates a 3D styling pipeline using FastAPI, Blender, and PyTorch Neural Style Transfer.

## Prerequisite
- **Blender**: Must be installed and accessible globally via `blender` command, or updated in `app/main.py` (`BLENDER_EXEC`).
- **Python 3.9+**

## Installation

1. Create virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(I have already initiated this for you)*

## Running the Server

Start the FastAPI backend:
```bash
.\venv\Scripts\python -m uvicorn app.main:app --reload
```

The server runs at `http://localhost:8000`.

## API Usage

### 1. Check Available Styles
**GET** `/styles`
Returns a list of valid style IDs.

### 2. Upload and Process
**POST** `/upload`
- **Form Data**:
    - `file`: The 3D model file (.obj, .glb, .gltf, .fbx).
    - `style_id`: ID of the style (e.g., `1`).

Returns a JSON response with `job_id` and download URL. **Note**: This process is synchronous and may take 1-3 minutes depending on hardware (for Style Transfer).

### 3. Download Result
**GET** `/download/{job_id}`
Returns the final stylized `.glb` file.

### 4. CLI Usage
You can run the pipeline directly from the command line without the server.
```bash
python cli.py demo_cube.obj 1 --output ./my_result
```
- `demo_cube.obj`: Included sample file.
- `1`: Style ID (1-9).


## Project Structure
- `app/`: FastAPI application code.
- `scripts/`: Blender python scripts for geometry processing.
- `styles/`: Reference style images.
- `temp/`: Temporary processing directory.
