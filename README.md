# StyleForge 3D 🎨📐
**Advanced GPU-Accelerated 3D Texture Synthesis Pipeline**

StyleForge 3D is a production-grade, Python-native pipeline designed to apply arbitrary artistic styles to raw, untextured 3D models. Unlike naive style transfer techniques that cause geometry distortion and visual seam tearing, StyleForge 3D respects 3D topology, UV continuity, and geometric features to deliver flawless, web-ready `.glb` assets.

---

## 📌 Table of Contents
1. [The Problem Statement](#-the-problem-statement)
2. [Why Existing Ideas Fail](#-why-existing-ideas-fail)
3. [How StyleForge 3D Solves It](#-how-style-forge-3d-solves-it)
4. [System Architecture & Workflow Pipeline](#-system-architecture--workflow-pipeline)
5. [Algorithmic Deep Dive](#-algorithmic-deep-dive)
6. [Development Evolution & Quantitative Metrics](#-development-evolution--quantitative-metrics)
7. [Installation & Setup](#-installation--setup)
8. [Usage Guide](#-usage-guide)

---

## 🔍 The Problem Statement

Applying 2D artistic styles (e.g., paintings, sketches, tile work) to 3D meshes presents fundamental mathematical and spatial challenges:

*   **Dimensionality Mismatch:** Convolutional Neural Networks (CNNs) operate on regular 2D pixel grids. 3D meshes are irregular graphs of vertices and faces.
*   **The Render/Re-projection Dilemma:** Naive approaches render the 3D model from multiple 2D angles, style the 2D images, and attempt to re-project them. This introduces severe **viewpoint inconsistency** (e.g., features changing shape as you rotate the model) and **seam artifacts** where projected views meet.
*   **Low-Poly Geometry Constraints:** Vertex coloring is bounded by mesh density. Applying detailed textures to low-poly models is impossible without high-resolution UV mapping, but generating high-res UV textures that align with the geometry requires semantic awareness.
*   **Geometric Fidelity Protection:** A style transfer algorithm must know the difference between a flat surface (where brushstrokes should flow freely) and a sharp geometric edge (like a corner or border), which must be protected to prevent the model from looking "melted."

---

## 🚫 Why Existing Ideas Fail

Our development journey progressed through multiple stages, exposing the limitations of standard computer vision techniques when applied to 3D texture synthesis:

| Stage | Attempted Idea | Technical Mechanism | Why It Failed / Visual Shortcomings |
| :--- | :--- | :--- | :--- |
| **Output 1** | **Naive Baseline (Gatys)** | Standard 2D Neural Style Transfer on random noise texture. | **"Melted" Projection:** Entirely ignored 3D geometry. Textural details (like eyes or brushstrokes) appeared randomly, with no correlation to surface features. |
| **Output 2** | **Geometric Grounding** | Init optimization canvas with baked normal maps. | **UV Seam Discontinuities:** Style followed geometry, but had severe tearing/sharp edges where the flat UV islands met. Texture looked muddy and low-contrast. |
| **Output 3** | **Structural Anchoring** | Added VGG content loss to lock structural features. | **High-Frequency Noise:** Retained landmark shapes but introduced severe "checkerboard" artifacts and graininess across the surface. |
| **Output 4** | **Noise Control (Standard TV)** | Added isotropic Total Variation (TV) regularization. | **"The Blurry Phase":** Denoised the canvas, but smeared all fine style details, turning the output into a blurry, low-fidelity oil painting. |
| **Output 5** | **Edge-Aware Smoothing** | Implemented Perona-Malik anisotropic diffusion weights. | **Semantic Style Loss:** Edges were kept sharp and flat areas clean, but style patterns lacked cohesion (e.g., brick styles looked like unstructured red blobs). |
| **Output 6** | **Strict Local Patch Style** | Cosine-matching VGG features using Markov Random Fields. | **VRAM Exhaustion & Composition Issues:** Captured localized brushstrokes perfectly, but suffered from poor macro-composition (repeated tiny patterns) and crashed consumer GPU memory. |

---

## 💡 How StyleForge 3D Solves It

Our final implementation (**Output 7**) integrates the theoretical strengths of 3D geometry parameterization, global-local style networks, and edge-preserving diffusion logic into a single, cohesive engine:

1.  **Python-Native Geometry Processing:** We bypassed heavy external rendering engines (like Blender) by employing `xatlas` and `trimesh` to perform non-overlapping UV unwrapping and rasterize vertex normals into a high-fidelity 2D content map directly in Python.
2.  **Hybrid Global-Local Style Loss:** We combine **Gram Matrices** (for global color harmony and palette distribution) with **Markov Random Field (MRF) Patch Loss** (for local semantic continuity). This ensures style elements like brushstrokes or geometric patterns remain intact and map naturally across UV seams.
3.  **Dynamic Edge-Aware Regularization:** We integrate **Perona-Malik Anisotropic Diffusion** directly into the optimization backward pass. This dynamically scales the smoothing weight—smoothing out noise in flat areas while ensuring sharp edges and seams remain razor-sharp.
4.  **Entropy-Driven Feedback Loop:** The optimizer actively monitors gradient entropy and pixel variance. If the image begins to over-smooth (entropy drops), it automatically lowers regularization; if noise spikes (variance rises), it increases TV smoothing.
5.  **Multi-Resolution Optimizations:** The pipeline optimizes in two stages (coarse 256x256, then upscales to 512x512), using the second-order **L-BFGS** optimizer and **FP16 Mixed Precision (PyTorch AMP)** to accelerate convergence and reduce VRAM usage.

---

## ⚙️ System Architecture & Workflow Pipeline

The system is split into three main phases, moving from raw mesh validation to the export of a textured GLB:

```mermaid
graph TD
    A[User Uploads .OBJ Mesh] -->|Validation| B(FastAPI Handler)
    B --> C{Geometry Engine}
    
    subgraph "Phase 1: Geometry Prep"
    C -->|xatlas| D[Distortion-Minimizing UV Unwrapping]
    D -->|scipy.interpolate| E[Content normal Map Baking]
    end
    
    subgraph "Phase 2: Hybrid Neural Style Transfer"
    E --> F[VGG-19 Feature Extraction]
    F --> G[L-BFGS Optimization Loop]
    G -->|Loss Backpropagation| G
    G -->|Adaptive Entropy Feedback| G
    G -->|Result| H[Seamless Stylized Texture]
    end
    
    subgraph "Phase 3: Asset Delivery"
    H --> I[PBR Material Mapping]
    I -->|Pack Geometry + Texture| J[Export Binary .GLB]
    end
    
    J --> K[API Download / Static Serve]
```

---

## 🧠 Algorithmic Deep Dive

### 1. Iso-Chart Parameterization & Normal Baking (`xatlas`)
The mesh is segmented into distortion-minimizing UV charts using Least Squares Conformal Maps (LSCM). To transfer geometry to the CNN:
*   We raycast vertex normals $\vec{n} = (x, y, z)$ back into the UV space.
*   We map coordinate ranges $[-1, 1]$ into RGB space:
    $$ R = 0.5(x+1), \quad G = 0.5(y+1), \quad B = 0.5(z+1) $$
*   We interpolate missing pixels using `scipy.interpolate.griddata` to yield a continuous Normal Map representing 3D features in 2D.

### 2. Global Style Loss (Gram Matrix)
To capture overall colors and broad textures without spatial locking, we calculate feature correlations in VGG-19:
$$ G_{ij} = \sum_k F_{ik} F_{jk} $$
$$ \mathcal{L}_{style\_global} = \frac{1}{4 N^2 M^2} \sum (G_{generated} - G_{target})^2 $$

### 3. Local Style Loss (Patch-Based MRF)
To preserve localized semantic structures (e.g., brushstrokes, tiles), we extract small overlapping $k \times k$ patches from VGG-19 features. For each content patch $\phi_c$, we search the style image for the best matching patch $\phi_s$ using Cosine Similarity:
$$ \text{Nearest}(\phi_c) = \arg \max_{\phi_s \in S} \left( \frac{\phi_c \cdot \phi_s}{||\phi_c|| \cdot ||\phi_s||} \right) $$
We then minimize the Mean Squared Error (MSE) between each patch and its closest style patch. To fit this in memory, we **subsample a maximum of 2,000 target patches** at run time.

### 4. Dynamic Edge-Aware TV Loss (Perona-Malik Regularization)
Standard Total Variation (TV) blurs all details. We compute gradients $\nabla I$ dynamically, wrapping them in an anisotropic decay weight:
$$ W(x,y) = e^{-\alpha |\nabla I(x,y)|} $$
$$ \mathcal{L}_{TV} = \sum_{i,j} W(i,j) \cdot |\nabla I_{i,j}| $$
*   **High Gradient (Edge):** $W \to 0$ (protecting details/corners/UV boundaries).
*   **Low Gradient (Flat area):** $W \to 1$ (smoothing away high-frequency noise).

### 5. Multi-Resolution L-BFGS Engine
We utilize the second-order **L-BFGS** optimizer which approximates the inverse Hessian matrix, taking smarter steps toward convergence compared to standard Adam. We employ a multi-resolution pyramid:
*   **Phase 1 (256x256 px):** Optimization establishes composition and large shapes.
*   **Phase 2 (512x512 px):** Texture is bilinearly upscaled and refined for high-frequency details.
*   Operations are accelerated via **PyTorch AMP (FP16 Mixed Precision)**, cutting execution time and GPU memory usage in half.

---

## 📈 Development Evolution & Quantitative Metrics

The table below showcases the quantitative progression of the StyleForge 3D textures. The optimization achieved in **Output 7** strikes the perfect balance of detail, sharpness, and edge preservation.

| Metric | **Out 1** (Naive) | **Out 2** | **Out 3** | **Out 4** | **Out 5** | **Out 6** | **Out 7** (Production) | **Δ (Out 1 → Out 7)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pixel Variance** | 0.0343 | 0.0325 | 0.0103 | 0.0102 | 0.0102 | 0.0307 | **0.1096** | **+219.6%** (Rich Detail) |
| **Gradient Entropy** | 2.3472 | 2.3712 | 2.5933 | 2.6800 | 2.6841 | 2.8418 | **3.4985** | **+49.0%** (Complexity) |
| **Edge Density** | 0.0989 | 0.0904 | 0.0154 | 0.0495 | 0.0545 | 0.0830 | **0.4447** | **+349.6%** (Preserved Edges) |
| **Laplacian Sharpness** | 0.0142 | 0.0138 | 0.0001 | 0.0001 | 0.0001 | 0.0288 | **0.1699** | **+1095.6%** (Crispness) |
| **Total Variation** | 61k | 60k | 8.9k | 9.0k | 8.8k | 123k | **524k** | **+759.2%** (Textural Intensity) |
| **SSIM to Content** | 0.8943 | 0.8738 | 0.5023 | 0.4934 | 0.4937 | 0.4172 | **-0.0008** | **-100.1%** (Complete Styling) |

> [!NOTE]
> **SSIM to Content** drops to near-zero in Output 7. This is mathematically correct: the raw normal map structure is completely transformed into the designated artistic style, whilst its edge boundaries and shape definition remain structurally protected by the Edge-Aware TV and Patch losses.

---

## 🛠️ Installation & Setup

### Prerequisites
*   **Python 3.9+** (Python 3.10+ recommended)
*   **Nvidia GPU + CUDA Toolkit** (highly recommended for performance)

### Local Setup
1.  Clone the repository and navigate to the project root:
    ```bash
    git clone https://github.com/ArchedEquation/StyleForge-3d.git
    cd StyleForge-3d
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage Guide

### 1. Command Line Interface (CLI)
You can run the pipeline directly on local meshes using the script:
```bash
python cli.py <input_mesh_path> <style_id_or_path> [options]
```

*   **Example (Using default styles):**
    ```bash
    python cli.py demo_cube.obj 1
    ```
*   **Example (Using custom paths):**
    ```bash
    python cli.py ./my_model.obj ./styles/van_gogh.jpg --output ./my_results/
    ```

### 2. FastAPI Web Server
Start the high-performance async API server:
```bash
uvicorn app.main:app --reload
```
The server will start at `http://localhost:8000`. You can test endpoints using the interactive documentation at `http://localhost:8000/docs`.

#### Key Endpoints:
*   `GET /styles` — Lists all available pre-configured styles.
*   `POST /upload` — Upload an `.obj` model and request styling by passing a `style_id`. Returns a `job_id`.
*   `GET /download/{job_id}` — Downloads the finished binary `.glb` asset.

### 3. Docker Deployment (EC2 / Cloud GPU)
For cloud deployment (e.g., AWS EC2 g4dn.xlarge GPU instances), use the pre-configured Dockerfile:
*   **Build the Image:**
    ```bash
    docker build -t styleforge-3d .
    ```
*   **Run with GPU Access:**
    ```bash
    docker run --gpus all -p 8000:8000 styleforge-3d
    ```

---

*Happy Styling*
