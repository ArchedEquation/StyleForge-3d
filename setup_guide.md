# StyleForge 3D - Setup Guide

This guide details how to set up, install, and run the StyleForge 3D environment on your local machine.

## 1. Prerequisites
- **Operating System**: Windows, macOS, or Linux. (Windows 10/11 recommended).
- **Python**: Version 3.9 or higher.
- **Hardware**: An NVIDIA GPU is **highly recommended** for Neural Style Transfer speeds. Running on CPU is possible but will be significantly slower (minutes vs seconds).
- **Git**: To clone the repository.

## 2. Installation

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd styleforge-3d
```

### Step 2: Create a Virtual Environment
It is best practice to run Python projects in an isolated environment.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install the required libraries, including PyTorch, FastAPI, Trimesh, and others.

```bash
pip install -r requirements.txt
```

> **Note on PyTorch (CUDA):**
> If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch. The default `pip install` might install the CPU version depending on your system. You can verify or force install from [pytorch.org](https://pytorch.org/), e.g.:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## 3. Usage

There are two ways to use StyleForge 3D: via the **Command Line Interface (CLI)** or the **Web API**.

### A. Command Line Interface (CLI)
Best for quick testing or batch processing locally.

**Syntax:**
```bash
python cli.py <input_model_path> <style_id> --output <output_folder>
```

**Example (Test with included demo):**
```bash
python cli.py demo_cube.obj 1 --output ./my_result
```
*   `demo_cube.obj`: A simple sample file included in the repo.
*   `1`: Corresponds to the style image ID (e.g., `styles/style-1.jpg`).
*   The result will be saved as `styleforge_result.glb` in the output folder.

### B. Web API Server
Best for integrating with frontends or other applications.

**Start the Server:**
```bash
python -m uvicorn app.main:app --reload
```

**Access the API:**
1.  Open your browser to `http://localhost:8000/docs`.
2.  Use the **Swagger UI** to interact with the endpoints:
    *   **GET /styles**: See available styles.
    *   **POST /upload**: Upload a `.obj` or `.glb` file and select a style ID.
    *   **GET /download/{job_id}**: Download your stylized model.

## 4. Troubleshooting

**1. "RuntimeError: No CUDA GPUs are available"**
*   The system defaulted to CPU. This is fine, but processing will take longer. If you have a GPU, reinstall PyTorch with CUDA support (see Step 3).

**2. "ValueError: different number of values and points"**
*   This indicates an issue with the UV process mismatching vertices. Ensure you have the latest code, as we applied a fix to `trimesh` loading (`process=False`) to handle XAtlas seams correctly.

**3. "Style not found"**
*   Ensure the `styles/` directory exists and contains images named `style-1.jpg`, `style-2.jpg`, etc.

## 5. Development Notes
*   **Pipeline Logic**: mostly contained in `app/pipeline.py`.
*   **Style Transfer**: core logic in `app/style_transfer.py`.
*   **Mesh Scripts**: helper scripts in `scripts/` folder using `trimesh` and `xatlas`.
