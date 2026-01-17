# StyleForge 3D - Model & Algorithm Details

This document outlines the technical details of the Neural Style Transfer (NST) model implemented in StyleForge 3D.

## 1. Core Algorithm
The implementation is based on the seminal paper **"A Neural Algorithm of Artistic Style"** by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

It uses a pre-trained **Convolutional Neural Network (CNN)** to separate and recombine the *content* of one image with the *style* of another.

## 2. Model Architecture
-   **Backbone**: **VGG19** (Visual Geometry Group, 19 layers).
-   **Weights**: Pre-trained on **ImageNet**.
-   **State**: The model is set to `eval()` mode, freezing all weights. We do *not* train the network itself; instead, we optimize the *input image pixels*.

## 3. Loss Functions
The generated image is optimized to minimize a weighted combination of three loss components:

### A. Content Loss
Ensures the result retains the structural shapes of the original 3D render.
-   **Layer**: `conv_4` (block 4, conv 2).
-   **Metric**: Mean Squared Error (MSE) between the feature maps of the content image and the generated image.
-   **Weight**: `1000` (High weight chosen to strictly preserve 3D geometry outlines).

### B. Style Loss
Ensures the result captures the textures and brushstrokes of the style reference.
-   **Layers**: `conv_1`, `conv_2`, `conv_3`, `conv_4`, `conv_5`.
-   **Metric**: MSE between the **Gram Matrices** (feature correlations) of the style image and the generated image.
-   **Weight**: `100,000`.

### C. Total Variation (TV) Loss
Reduces high-frequency noise and checkerboard artifacts, encouraging spatial smoothness.
-   **Weight**: `1e-6`.

## 4. Optimization Process
Instead of a single forward pass (like Fast-NST methods), this pipeline uses an iterative optimization process for higher quality.

-   **Optimizer**: **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno). This optimizer is generally slower but converges to higher quality results than standard SGD/Adam for style transfer.
-   **Initialization**: The canvas is initialized with the **Content Image** (not random noise). This is critical for 3D mappings to ensure UV charts and internal geometry remain consistent.
-   **Steps**: 150 iterations.
-   **Resolution**: 512x512 pixels.

## 5. Typical Workflow
1.  **3D Bake**: The 3D model is unwrapped, and its surface normals/geometry are baked into a 2D "Content Map".
2.  **Optimization**: The VGG19 network extracts features from the Content Map and the Style Image. The optimizer adjusts pixel values to match the target content and style representations.
3.  **texturing**: The resulting stylized image is applied back to the 3D model as a texture.
