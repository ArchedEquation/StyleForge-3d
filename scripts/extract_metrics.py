"""
StyleForge 3D - Output Metrics Extraction & Comparative Analysis Script
========================================================================
This script analyses all 9 output directories, extracts image quality metrics 
from each stylized texture (processed_style.png), and produces a comparative 
report showing which method was used at each stage and how it improved upon 
its predecessor.

Metrics Computed:
-----------------
1. Mean Pixel Value         - Average brightness/illumination of texture
2. Pixel Variance           - Measures detail richness (low=flat, high=detailed)
3. Gradient Entropy         - Shannon entropy of gradient histogram (texture complexity)
4. Edge Density             - % of pixels classified as strong edges (Sobel)
5. SSIM vs Content Map      - Structural Similarity to original normal map (content preservation)
6. Color Channel Divergence - Inter-channel variance (high=colorful, low=grayscale)
7. Anisotropy Score         - Ratio of directional gradients (1=isotropic, >1=directional strokes)
8. File Size (KB)           - Compressed PNG size as proxy for information density
9. Total Variation          - Sum of absolute pixel differences (smoothness measure)
10. BRISQUE-like Sharpness  - Laplacian variance (higher = sharper)
"""

import os
import sys
import numpy as np
from PIL import Image
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_FILE = os.path.join(BASE_DIR, "metrics_analysis_report.md")

# The 7 evolutionary outputs + extra runs (up to 9)
OUTPUT_FOLDERS = [f"output{i}" for i in range(1, 10)]

# Method used at each output stage
METHODS = {
    "output1": {
        "name": "Naive Baseline (Gatys et al.)",
        "technique": "Standard Neural Style Transfer on random noise",
        "components": ["VGG-19 Feature Extraction", "Gram Matrix Style Loss", "MSE Content Loss", "Adam Optimizer"],
        "paper": "Gatys, Ecker & Bethge (2016) - 'Image Style Transfer Using Convolutional Neural Networks'",
        "link": "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf",
        "improvement": "Established the baseline. No geometric awareness — style is applied blindly to a 2D noise canvas."
    },
    "output2": {
        "name": "Geometric Grounding (Content Map Init)",
        "technique": "Content Map Baking (Surface Normals → RGB) + Content Initialization",
        "components": ["VGG-19", "Gram Matrix", "Normal Map Baking (xatlas + scipy.griddata)", "Content Initialization"],
        "paper": "Czerkawski et al. (2020) - 'Content initialization for style transfer of 3D textures'",
        "link": "https://arxiv.org/abs/2006.09415",
        "improvement": "Over Output 1: Replaced random noise initialization with baked normal maps. The optimizer starts from a geometrically meaningful state, so style follows curvature instead of random placement. SSIM to content should increase significantly."
    },
    "output3": {
        "name": "Structure Preservation (Content Loss Anchoring)",
        "technique": "Weighted Content Loss targeting structural features at conv4_2",
        "components": ["VGG-19", "Gram Matrix", "Structural Content Loss (conv3_2 + conv4_2)", "Sobel Edge Masking"],
        "paper": "Johnson, Alahi & Fei-Fei (2016) - 'Perceptual Losses for Real-Time Style Transfer'",
        "link": "https://arxiv.org/abs/1603.08155",
        "improvement": "Over Output 2: Added explicit content loss that penalizes deviation from geometric features. Edge Density should increase as edges become sharper. However, introduces high-frequency noise (Gradient Entropy may spike)."
    },
    "output4": {
        "name": "Noise Control (Standard Total Variation)",
        "technique": "Standard isotropic Total Variation regularization",
        "components": ["VGG-19", "Gram Matrix", "Content Loss", "Total Variation Loss: Σ|I(x+1)-I(x)|"],
        "paper": "Rudin, Osher & Fatemi (1992) - 'Nonlinear total variation based noise removal algorithms'",
        "link": "https://doi.org/10.1016/0167-2789(92)90242-F",
        "improvement": "Over Output 3: TV loss suppresses high-frequency noise. Gradient Entropy drops (smoother). But Total Variation itself drops too aggressively — edges are blurred. Edge Density decreases. This is the 'blurry phase'."
    },
    "output5": {
        "name": "Detail Enhancement (Perona-Malik Edge-Aware Diffusion)",
        "technique": "Anisotropic Diffusion: W = exp(-α|∇I|) weighted TV",
        "components": ["VGG-19", "Gram Matrix", "Content Loss", "Edge-Aware TV (Perona-Malik)", "Dynamic α=5.0"],
        "paper": "Perona & Malik (1990) - 'Scale-space and edge detection using anisotropic diffusion'",
        "link": "https://ieeexplore.ieee.org/document/56205",
        "improvement": "Over Output 4: THE BREAKTHROUGH. Replaces isotropic TV with gradient-adaptive weights. Smooth areas are aggressively denoised (W→1), edges are preserved (W→0). Expect: Edge Density recovers while flat-area noise stays low. Sharpness (Laplacian variance) increases dramatically."
    },
    "output6": {
        "name": "Semantic Coherence (MRF Patch-Based Style Loss)",
        "technique": "Markov Random Field neural patch matching via cosine similarity",
        "components": ["VGG-19", "Gram Matrix + Patch-MRF Loss", "Content Loss", "Edge-Aware TV", "torch.nn.functional.unfold", "Cosine Similarity NN Search"],
        "paper": "Li & Wand (2016) - 'Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis'",
        "link": "https://arxiv.org/abs/1601.04589",
        "improvement": "Over Output 5: Gram matrices treat textures as unordered 'bags of features'. MRF patches preserve local spatial arrangement — strokes, slabs, and gradients stay intact. Anisotropy Score should reflect more directional texture. Color Divergence may increase as the style's color structure is better preserved."
    },
    "output7": {
        "name": "Final Polish (Multi-Resolution Coarse-to-Fine)",
        "technique": "Pyramid optimization: 256px→512px with L-BFGS + entropy/variance feedback",
        "components": ["VGG-19", "Gram + Patch-MRF", "Content Loss", "Edge-Aware TV", "L-BFGS Optimizer", "Multi-Resolution (256→512)", "Entropy Monitor", "Variance Monitor"],
        "paper": "Gatys et al. (2017) - 'Controlling Perceptual Factors in Neural Style Transfer' + Nocedal (1980) - 'Updating quasi-Newton matrices with limited storage'",
        "link": "https://arxiv.org/abs/1611.07865",
        "improvement": "Over Output 6: Multi-resolution locks composition at low-res, then refines details at high-res. L-BFGS provides second-order convergence (smarter steps). Adaptive feedback (entropy<2.5 → reduce TV; variance>0.15 → increase TV) prevents over-smoothing and noise spikes. This is PRODUCTION READY. All metrics should be at their best balance."
    },
    "output8": {
        "name": "Extended Run / Alternative Style (Post-Production)",
        "technique": "Full pipeline on different style image",
        "components": ["Full Output 7 Architecture"],
        "paper": "Same as Output 7",
        "link": "https://arxiv.org/abs/1611.07865",
        "improvement": "Applies the final architecture to a different style, validating generalization. Metrics should be comparable to Output 7 with style-dependent variations."
    },
    "output9": {
        "name": "Extended Run / Alternative Style 2 (Post-Production)",
        "technique": "Full pipeline on another style image",
        "components": ["Full Output 7 Architecture"],
        "paper": "Same as Output 7",
        "link": "https://arxiv.org/abs/1611.07865",
        "improvement": "Further generalization test. Confirms the pipeline is style-agnostic."
    }
}


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def load_image_as_array(path):
    """Load image and return as float32 numpy array normalized to [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_mean_pixel(img):
    """Average pixel intensity across all channels."""
    return float(np.mean(img))


def compute_variance(img):
    """Global pixel variance — measures detail/information content."""
    return float(np.var(img))


def compute_gradient_entropy(img):
    """
    Shannon entropy of the gradient magnitude histogram.
    Higher entropy = more complex texture.
    Based on: Shannon (1948) - 'A Mathematical Theory of Communication'
    """
    gray = np.mean(img, axis=2)
    dx = gray[:, :-1] - gray[:, 1:]
    dy = gray[:-1, :] - gray[1:, :]
    # Crop to matching shapes
    dx = dx[:-1, :]
    dy = dy[:, :-1]
    mag = np.sqrt(dx**2 + dy**2 + 1e-8)
    
    # Histogram (100 bins)
    hist, _ = np.histogram(mag.flatten(), bins=100, range=(0, mag.max() + 1e-8))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(entropy)


def compute_edge_density(img, threshold=0.15):
    """
    Fraction of pixels with Sobel gradient magnitude above threshold.
    Uses Sobel operator: Gx, Gy kernels.
    Reference: Sobel & Feldman (1968) - 'A 3x3 Isotropic Gradient Operator for Image Processing'
    """
    gray = np.mean(img, axis=2)
    # Sobel X
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    from scipy.ndimage import convolve
    gx = convolve(gray, kx)
    gy = convolve(gray, ky)
    mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize
    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val
    
    edge_pixels = np.sum(mag > threshold)
    total_pixels = gray.size
    return float(edge_pixels / total_pixels)


def compute_ssim_to_content(style_img, content_img):
    """
    Structural Similarity Index (SSIM) between stylized output and content map.
    Measures how well the 3D structure is preserved after style transfer.
    Reference: Wang et al. (2004) - 'Image quality assessment: from error visibility to structural similarity'
    Link: https://ieeexplore.ieee.org/document/1284395
    """
    # Simplified SSIM (luminance, contrast, structure)
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Resize content to match style if needed
    from PIL import Image as PILImage
    if style_img.shape != content_img.shape:
        h, w = style_img.shape[:2]
        content_pil = PILImage.fromarray((content_img * 255).astype(np.uint8))
        content_pil = content_pil.resize((w, h), PILImage.BILINEAR)
        content_img = np.array(content_pil, dtype=np.float32) / 255.0
    
    mu_x = np.mean(style_img)
    mu_y = np.mean(content_img)
    sigma_x = np.var(style_img)
    sigma_y = np.var(content_img)
    sigma_xy = np.mean((style_img - mu_x) * (content_img - mu_y))
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return float(ssim)


def compute_color_divergence(img):
    """
    Mean absolute inter-channel difference.
    High = colorful and rich palette; Low = grayscale/monochrome.
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    div = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    return float(div)


def compute_anisotropy(img):
    """
    Ratio of horizontal to vertical gradient energy.
    Ratio ≈ 1.0 means isotropic texture; >1 or <1 means directional strokes.
    """
    gray = np.mean(img, axis=2)
    dx = gray[:, :-1] - gray[:, 1:]
    dy = gray[:-1, :] - gray[1:, :]
    
    energy_x = np.sum(dx**2)
    energy_y = np.sum(dy**2)
    
    if energy_y < 1e-8:
        return 1.0
    return float(energy_x / energy_y)


def compute_total_variation(img):
    """
    Standard Total Variation: Σ|I(x+1,y) - I(x,y)| + Σ|I(x,y+1) - I(x,y)|
    Lower = smoother image; Higher = more detail/noise.
    Reference: Rudin, Osher & Fatemi (1992)
    """
    dx = np.abs(img[:, :-1, :] - img[:, 1:, :])
    dy = np.abs(img[:-1, :, :] - img[1:, :, :])
    return float(np.sum(dx) + np.sum(dy))


def compute_laplacian_sharpness(img):
    """
    Variance of Laplacian — standard no-reference sharpness metric.
    Higher = sharper. Acts as a BRISQUE-like quality proxy.
    Reference: Pech-Pacheco et al. (2000) - 'Diatom autofocusing in brightfield microscopy'
    """
    gray = np.mean(img, axis=2)
    # Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    lap = convolve(gray, kernel)
    return float(np.var(lap))


def get_file_size_kb(path):
    """File size in KB."""
    return os.path.getsize(path) / 1024.0


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis():
    print("=" * 80)
    print("  StyleForge 3D — Output Metrics Extraction & Comparative Analysis")
    print("=" * 80)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    all_results = {}
    
    for folder_name in OUTPUT_FOLDERS:
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        style_img_path = os.path.join(folder_path, "processed_style.png")
        content_img_path = os.path.join(folder_path, "processed_content.png")
        
        if not os.path.exists(style_img_path):
            print(f"\n  [SKIP] {folder_name}: No processed_style.png found.")
            continue
        
        print(f"\n  Analysing {folder_name}...")
        
        # Load images
        style_img = load_image_as_array(style_img_path)
        content_img = load_image_as_array(content_img_path) if os.path.exists(content_img_path) else None
        
        # Compute metrics
        metrics = {
            "resolution": f"{style_img.shape[1]}x{style_img.shape[0]}",
            "file_size_kb": round(get_file_size_kb(style_img_path), 2),
            "mean_pixel": round(compute_mean_pixel(style_img), 5),
            "pixel_variance": round(compute_variance(style_img), 6),
            "gradient_entropy": round(compute_gradient_entropy(style_img), 4),
            "edge_density": round(compute_edge_density(style_img), 4),
            "color_divergence": round(compute_color_divergence(style_img), 5),
            "anisotropy_score": round(compute_anisotropy(style_img), 4),
            "total_variation": round(compute_total_variation(style_img), 2),
            "laplacian_sharpness": round(compute_laplacian_sharpness(style_img), 8),
            "ssim_to_content": round(compute_ssim_to_content(style_img, content_img), 5) if content_img is not None else "N/A"
        }
        
        all_results[folder_name] = metrics
        
        # Print summary
        print(f"    Resolution:          {metrics['resolution']}")
        print(f"    File Size:           {metrics['file_size_kb']:.2f} KB")
        print(f"    Mean Pixel:          {metrics['mean_pixel']:.5f}")
        print(f"    Pixel Variance:      {metrics['pixel_variance']:.6f}")
        print(f"    Gradient Entropy:    {metrics['gradient_entropy']:.4f}")
        print(f"    Edge Density:        {metrics['edge_density']:.4f}")
        print(f"    Color Divergence:    {metrics['color_divergence']:.5f}")
        print(f"    Anisotropy:          {metrics['anisotropy_score']:.4f}")
        print(f"    Total Variation:     {metrics['total_variation']:.2f}")
        print(f"    Laplacian Sharpness: {metrics['laplacian_sharpness']:.8f}")
        print(f"    SSIM to Content:     {metrics['ssim_to_content']}")
    
    # ========================================================================
    # GENERATE MARKDOWN REPORT
    # ========================================================================
    print("\n\n  Generating Markdown Report...")
    
    report_lines = []
    report_lines.append("# StyleForge 3D — Output Metrics & Comparative Analysis Report")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")
    
    # ---- METRICS TABLE ----
    report_lines.append("## 📊 Extracted Metrics Summary\n")
    
    metric_keys = ["file_size_kb", "mean_pixel", "pixel_variance", "gradient_entropy",
                    "edge_density", "color_divergence", "anisotropy_score",
                    "total_variation", "laplacian_sharpness", "ssim_to_content"]
    metric_labels = {
        "file_size_kb": "File Size (KB)",
        "mean_pixel": "Mean Pixel",
        "pixel_variance": "Pixel Variance",
        "gradient_entropy": "Gradient Entropy",
        "edge_density": "Edge Density",
        "color_divergence": "Color Divergence",
        "anisotropy_score": "Anisotropy",
        "total_variation": "Total Variation",
        "laplacian_sharpness": "Laplacian Sharpness",
        "ssim_to_content": "SSIM to Content"
    }
    
    available_outputs = [k for k in OUTPUT_FOLDERS if k in all_results]
    
    if not available_outputs:
        report_lines.append("*No output data found.*\n")
    else:
        # Build table
        header = "| Metric |" + "|".join([f" **{o.replace('output','Out ')}** " for o in available_outputs]) + "|"
        sep = "|:---|" + "|".join(["---:" for _ in available_outputs]) + "|"
        report_lines.append(header)
        report_lines.append(sep)
        
        for mk in metric_keys:
            label = metric_labels.get(mk, mk)
            row = f"| **{label}** |"
            for o in available_outputs:
                val = all_results[o].get(mk, "N/A")
                row += f" {val} |"
            report_lines.append(row)
        
        report_lines.append("")
    
    # ---- PER-OUTPUT DEEP DIVE ----
    report_lines.append("---\n")
    report_lines.append("## 🔬 Per-Output Method Analysis & Evolutionary Comparison\n")
    
    prev_output = None
    for output_key in available_outputs:
        info = METHODS.get(output_key, {})
        metrics = all_results[output_key]
        
        output_num = output_key.replace("output", "")
        report_lines.append(f"### Output {output_num}: {info.get('name', 'Unknown')}\n")
        
        # Method Card
        report_lines.append(f"| Property | Detail |")
        report_lines.append(f"|:---|:---|")
        report_lines.append(f"| **Technique** | {info.get('technique', 'N/A')} |")
        report_lines.append(f"| **Components** | {', '.join(info.get('components', []))} |")
        report_lines.append(f"| **Research Paper** | {info.get('paper', 'N/A')} |")
        report_lines.append(f"| **Paper Link** | [{info.get('link', '#')}]({info.get('link', '#')}) |")
        report_lines.append("")
        
        # Explanation
        report_lines.append(f"**Why this method?**")
        report_lines.append(f"> {info.get('improvement', 'N/A')}\n")
        
        # Key Metrics
        report_lines.append(f"**Key Metrics:**")
        report_lines.append(f"- Pixel Variance: `{metrics['pixel_variance']}` — " + 
                          ("⚠️ Very low (flat/collapsed)" if metrics['pixel_variance'] < 0.005 else 
                           "✅ Healthy detail range" if metrics['pixel_variance'] < 0.1 else 
                           "⚡ High variance (rich detail or noisy)"))
        report_lines.append(f"- Gradient Entropy: `{metrics['gradient_entropy']}` — " +
                          ("⚠️ Over-smoothed" if metrics['gradient_entropy'] < 2.5 else
                           "✅ Good complexity" if metrics['gradient_entropy'] < 4.0 else
                           "📈 Very complex texture"))
        report_lines.append(f"- Edge Density: `{metrics['edge_density']}` — " +
                          ("⚠️ Few edges (blurry)" if metrics['edge_density'] < 0.2 else
                           "✅ Good edge preservation" if metrics['edge_density'] < 0.6 else
                           "📐 Edge-heavy"))
        report_lines.append(f"- Laplacian Sharpness: `{metrics['laplacian_sharpness']}`")
        report_lines.append(f"- SSIM to Content: `{metrics['ssim_to_content']}`")
        report_lines.append("")
        
        # Comparison with previous
        if prev_output and prev_output in all_results:
            prev_metrics = all_results[prev_output]
            report_lines.append(f"**Δ Improvement over {prev_output.replace('output','Output ')}:**")
            
            # Calculate deltas
            for mk in ["pixel_variance", "gradient_entropy", "edge_density", "laplacian_sharpness"]:
                curr_val = metrics.get(mk, 0)
                prev_val = prev_metrics.get(mk, 0)
                if isinstance(curr_val, str) or isinstance(prev_val, str):
                    continue
                delta = curr_val - prev_val
                pct = (delta / prev_val * 100) if prev_val != 0 else 0
                arrow = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                report_lines.append(f"  - {metric_labels.get(mk, mk)}: {arrow} `{delta:+.6f}` ({pct:+.1f}%)")
            
            # SSIM comparison
            curr_ssim = metrics.get("ssim_to_content", "N/A")
            prev_ssim = prev_metrics.get("ssim_to_content", "N/A")
            if isinstance(curr_ssim, (int, float)) and isinstance(prev_ssim, (int, float)):
                ssim_delta = curr_ssim - prev_ssim
                ssim_arrow = "📈" if ssim_delta > 0 else "📉"
                report_lines.append(f"  - SSIM to Content: {ssim_arrow} `{ssim_delta:+.5f}` (structure {'better' if ssim_delta > 0 else 'worse'} preserved)")
            
            report_lines.append("")
        
        report_lines.append("---\n")
        prev_output = output_key
    
    # ---- RESEARCH REFERENCES ----
    report_lines.append("## 📚 Research References\n")
    report_lines.append("| # | Paper | Year | Used In | Link |")
    report_lines.append("|:--|:------|:-----|:--------|:-----|")
    
    refs = [
        ("1", "Gatys, Ecker & Bethge — Image Style Transfer Using CNNs", "2016", "Output 1 (Baseline)", "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf"),
        ("2", "Johnson, Alahi & Fei-Fei — Perceptual Losses for Real-Time Style Transfer", "2016", "Output 3 (Content Loss)", "https://arxiv.org/abs/1603.08155"),
        ("3", "Rudin, Osher & Fatemi — Nonlinear total variation based noise removal", "1992", "Output 4 (TV Loss)", "https://doi.org/10.1016/0167-2789(92)90242-F"),
        ("4", "Perona & Malik — Scale-space and edge detection using anisotropic diffusion", "1990", "Output 5 (Edge-Aware TV)", "https://ieeexplore.ieee.org/document/56205"),
        ("5", "Li & Wand — Combining MRFs and CNNs for Image Synthesis", "2016", "Output 6 (Patch Loss / MRF)", "https://arxiv.org/abs/1601.04589"),
        ("6", "Gatys et al. — Controlling Perceptual Factors in NST", "2017", "Output 7 (Multi-Resolution)", "https://arxiv.org/abs/1611.07865"),
        ("7", "Nocedal — Updating quasi-Newton matrices with limited storage (L-BFGS)", "1980", "Output 7 (Optimizer)", "https://doi.org/10.1090/S0025-5718-1980-0572855-7"),
        ("8", "Wang et al. — SSIM: Image Quality Assessment", "2004", "Metric: SSIM", "https://ieeexplore.ieee.org/document/1284395"),
        ("9", "Shannon — A Mathematical Theory of Communication", "1948", "Metric: Gradient Entropy", "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x"),
    ]
    for r in refs:
        report_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | [{r[4][:50]}...]({r[4]}) |")
    
    report_lines.append("")
    
    # ---- SUMMARY ----
    report_lines.append("---\n")
    report_lines.append("## 🏆 Summary: Why Output 7 is Superior\n")
    report_lines.append("| Feature | Standard NST (Output 1) | StyleForge 3D Final (Output 7) |")
    report_lines.append("|:--------|:------------------------|:-------------------------------|")
    report_lines.append("| **Initialization** | Random Noise | Content Map (Normal Maps) |")
    report_lines.append("| **Regularization** | None / Static TV | Dynamic Edge-Aware (Perona-Malik) |")
    report_lines.append("| **Style Matching** | Global Gram Only | Global Gram + Local MRF Patches |")
    report_lines.append("| **Edge Handling** | Zero Padding | Reflection Padding |")
    report_lines.append("| **Control** | Fixed Weights | Adaptive (Entropy + Variance Feedback) |")
    report_lines.append("| **Optimizer** | Adam (1st order) | L-BFGS (Quasi-Newton, 2nd order) |")
    report_lines.append("| **Resolution** | Single pass | Multi-Resolution (256→512) |")
    report_lines.append("")
    
    if "output1" in all_results and "output7" in all_results:
        o1 = all_results["output1"]
        o7 = all_results["output7"]
        report_lines.append("**Quantitative Improvement (Output 1 → Output 7):**\n")
        for mk in metric_keys:
            v1 = o1.get(mk, "N/A")
            v7 = o7.get(mk, "N/A")
            if isinstance(v1, (int, float)) and isinstance(v7, (int, float)) and v1 != 0:
                pct = (v7 - v1) / abs(v1) * 100
                report_lines.append(f"- **{metric_labels.get(mk, mk)}**: `{v1}` → `{v7}` ({pct:+.1f}%)")
            else:
                report_lines.append(f"- **{metric_labels.get(mk, mk)}**: `{v1}` → `{v7}`")
    
    report_lines.append("\n---\n*Report generated by `scripts/extract_metrics.py`*\n")
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\n  ✅ Report saved to: {RESULTS_FILE}")
    
    # Also save raw JSON
    json_path = os.path.join(BASE_DIR, "metrics_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✅ Raw metrics JSON saved to: {json_path}")
    
    print("\n" + "=" * 80)
    print("  Analysis Complete!")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    run_analysis()
