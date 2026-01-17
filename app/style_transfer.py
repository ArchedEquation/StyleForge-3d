import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import logging
import numpy as np

# Set up logging
logger = logging.getLogger("StyleTransferModule")
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512  # Output resolution

# --- Image Helpers ---
def get_image_loader(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

def image_loader(image_path, size=imsize):
    transform = get_image_loader(size)
    image = Image.open(image_path).convert('RGB')
    
    # Check for empty/uniform image
    np_img = np.array(image)
    if np.std(np_img) < 0.1:  
        logger.warning(f"Image {image_path} seems to have very low variance (uniform color?).")
    
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor, path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)

# --- Edge Detection ---
def get_edge_mask(img_tensor):
    """
    Computes a Sobel edge mask from the image tensor.
    img_tensor: [1, 3, H, W]
    Returns: [1, 1, H, W] float mask in [0, 1]
    """
    # Create kernels for Sobel
    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img_tensor.device).float().view(1, 1, 3, 3)
    k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img_tensor.device).float().view(1, 1, 3, 3)
    
    # Convert to grayscale for edge detection
    gray = img_tensor.mean(dim=1, keepdim=True) # [1, 1, H, W]
    
    # Padding to keep size
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    
    gx = F.conv2d(gray_padded, k_x)
    gy = F.conv2d(gray_padded, k_y)
    
    mag = torch.sqrt(gx**2 + gy**2)
    
    # Normalize
    max_val = mag.max()
    if max_val > 0:
        mag = mag / max_val
    
    # Enhance contrast: values near edges should be 1, others lower.
    # The prompt requests: "Strong content loss near edges, Weakened content loss on flat regions".
    # So we want high values at edges.
    
    # Maybe add a baseline so flat regions aren't 0 (which would mean NO content preservation).
    # Let's say baseline 0.2, edges 1.0.
    mag = torch.clamp(mag * 0.8 + 0.2, 0, 1) # Simple remapping
    
    return mag.detach()

# --- Loss Modules ---

class ContentLoss(nn.Module):
    def __init__(self, target, mask=None, weight=1.0):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.detach() if mask is not None else torch.ones_like(target).detach()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        # MSE weighted by mask
        self.loss = self.weight * F.mse_loss(input * self.mask, self.target * self.mask)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # use reshape instead of view to safely handle non-contiguous slices
    features = input.reshape(a * b, c * d)
    G = torch.mm(features, features.t())  
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature, layer_weight=1.0):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.layer_weight = layer_weight
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        # Weighted MSE
        self.loss = self.layer_weight * F.mse_loss(G, self.target)
        return input

class EdgeAwareTVLoss(nn.Module):
    def __init__(self, weight, edge_mask):
        super(EdgeAwareTVLoss, self).__init__()
        self.weight = weight
        # Invert edge mask: Low weight on edges (1), High weight on flat areas (0)
        # Actually, user wants: "Penalize smoothing only in low-gradient regions"
        # So weight should be high where edges are weak.
        # Mask is 1 at edges, 0 at flat. 
        # Weight map = (1 - mask)
        self.weight_map = (1.0 - edge_mask).detach()
        self.loss = 0

    def set_weight(self, w):
        self.weight = w

    def forward(self, input):
        # Calculate gradients
        h_x = input[:, :, :, :-1] - input[:, :, :, 1:]
        h_y = input[:, :, :-1, :] - input[:, :, 1:, :]
        
        # We need to align the weight map with the gradients. 
        # Gradients are 1 pixel smaller. Crop weight map? 
        # Or Just use 'same' padding logic implicitly?
        # Let's crop weight map to match gradient size for strict correctness.
        
        w_x = self.weight_map[:, :, :, 1:]
        w_y = self.weight_map[:, :, 1:, :]
        
        self.loss = self.weight * (
            torch.sum(torch.abs(h_x) * w_x) + 
            torch.sum(torch.abs(h_y) * w_y)
        )
        return self.loss

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Check if mean/std are already tensors
        if isinstance(mean, torch.Tensor):
             self.mean = mean.detach().clone().view(-1, 1, 1).to(device)
        else:
             self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
             
        if isinstance(std, torch.Tensor):
             self.std = std.detach().clone().view(-1, 1, 1).to(device)
        else:
             self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# --- Model Builder ---

class PatchStyleLoss(nn.Module):
    def __init__(self, target_feature, layer_weight=1.0):
        super(PatchStyleLoss, self).__init__()
        self.layer_weight = layer_weight
        self.loss = 0
        
        # create 2x2 grid patches
        H, W = target_feature.shape[-2:]
        h_mid, w_mid = H // 2, W // 2
        
        # We store the target grams for each quadrant
        # 0: TL, 1: TR, 2: BL, 3: BR
        self.targets = []
        self.slices = [
            (slice(None), slice(None), slice(0, h_mid), slice(0, w_mid)),
            (slice(None), slice(None), slice(0, h_mid), slice(w_mid, None)),
            (slice(None), slice(None), slice(h_mid, None), slice(0, w_mid)),
            (slice(None), slice(None), slice(h_mid, None), slice(w_mid, None))
        ]
        
        for sl in self.slices:
            self.targets.append(gram_matrix(target_feature[sl]).detach())

    def forward(self, input):
        total_loss = 0
        for i, sl in enumerate(self.slices):
            G = gram_matrix(input[sl])
            total_loss += F.mse_loss(G, self.targets[i])
            
        self.loss = self.layer_weight * (total_loss / 4.0)
        return input

# --- Model Builder ---

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               style_layers_config, content_layers_config):
    
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []
    
    # Calculate mask for Content Image
    content_edge_mask = get_edge_mask(content_img)
    
    # Build model
    model = nn.Sequential(normalization)

    i = 0 
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # Style Layers
        if name in style_layers_config:
            target_feature = model(style_img).detach()
            w = style_layers_config[name]
            
            # 1. Global Style Loss
            style_loss = StyleLoss(target_feature, layer_weight=w)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            
            # 2. Localized Patch Style Loss (Augmenting Global)
            # Apply only to mid-levels (conv_3, conv_5) where structure matters
            if name in ['conv_3', 'conv_5']:
                 patch_loss = PatchStyleLoss(target_feature, layer_weight=w * 0.5) # 50% weight of global
                 model.add_module("patch_style_loss_{}".format(i), patch_loss)
                 style_losses.append(patch_loss)

        # Content Layers
        if name in content_layers_config:
            target = model(content_img).detach()
            w = content_layers_config[name]
            
            # Upsample mask to current feature map size
            mask_resized = F.interpolate(content_edge_mask, size=target.shape[-2:], mode='nearest')
            
            content_loss = ContentLoss(target, mask=mask_resized, weight=w)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

    # Trim layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], PatchStyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

# --- Optimization Loop ---

def run_optimization(model, style_losses, content_losses, tv_loss_module, 
                     input_img, num_steps, 
                     style_weight, content_weight, 
                     tv_start, tv_end, phase_name="Optimization"):
    
    optimizer = optim.LBFGS([input_img])
    
    run = [0]
    
    # Store initial tv weight
    current_tv_w = tv_start
    
    while run[0] <= num_steps:
        def closure():
            nonlocal current_tv_w
            # Clamp
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Apply Global Weights
            style_score *= style_weight
            content_score *= content_weight
            
            # Adaptive TV Loss
            # 1. Decay based on progress
            progress = run[0] / num_steps
            base_tv = tv_start - (tv_start - tv_end) * progress
            
            # 2. Adaptive Adjustment based on Variance
            # If variance is dropping too low (over-smoothing), reduce TV pressure
            # If variance is high (noise), maintain or increase
            with torch.no_grad():
                var = torch.var(input_img).item()
                if var < 0.005: 
                    # Dangerously smooth, reduce TV to let details form
                    current_tv_w = base_tv * 0.5 
                elif var > 0.15:
                    # Very noisy
                    current_tv_w = base_tv * 1.2
                else:
                    current_tv_w = base_tv
            
            tv_loss_module.set_weight(current_tv_w)
            tv_score = tv_loss_module(input_img)

            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                logger.info(f"[{phase_name}] Step {run[0]}/{num_steps}: Style: {style_score.item():.2f} "
                            f"Content: {content_score.item():.2f} TV: {tv_score.item():.4f} "
                            f"(w={current_tv_w:.5f}, var={var:.4f}) Total: {loss.item():.2f}")
                
            return loss

        optimizer.step(closure)
    
    # Final clamp
    with torch.no_grad():
        input_img.clamp_(0, 1)

def run_style_transfer(content_path, style_path, output_path, num_steps=300): 
    # Not using num_steps because we have phases.
    logger.info("Initializing Advanced 3D Style Transfer...")

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Configuration mapped to VGG19 Layer Names (1-indexed Conv layers)
    # conv_1 (1_1), conv_2 (1_2), conv_3 (2_1), conv_4 (2_2), conv_5 (3_1), conv_9 (4_1), conv_13 (5_1)
    
    # User Request: conv_1: 1.0, conv_2: 0.8...
    # Assuming user means block indices or standard 1-5 mapping.
    # Standard Gatys mapping usually is: 
    # conv1_1 (conv_1), conv2_1 (conv_3), conv3_1 (conv_5), conv4_1 (conv_9), conv5_1 (conv_13)
    # Let's map "conv_1" to "conv_1", "conv_2" to "conv_3", etc. to match standard NST "5 layers".
    
    # Updated Style Config: Emphasize mid-level (conv3_1, conv4_1 -> conv_5, conv_9)
    # Reduce early layer noise (conv_1).
    style_config = {
        'conv_1': 0.5,  # conv1_1 (Reduced)
        'conv_3': 0.6,  # conv2_1
        'conv_5': 1.0,  # conv3_1 (Emphasized)
        'conv_9': 1.0,  # conv4_1 (Emphasized)
        'conv_13': 0.5  # conv5_1
    }
    
    # Content: conv_3_2 -> conv_6 ? (1(1_1)+1(1_2) + 1(2_1)+1(2_2) + 1(3_1) + 1(3_2) = 6th conv)
    # conv_4_2 -> conv_10 ? (6 + 3_3, 3_4, 4_1, 4_2 = 10)
    content_config = {
        'conv_6': 0.7,   # conv3_2
        'conv_10': 0.3   # conv4_2
    }

    # --- Multi-Resolution Optimization ---
    
    resolutions = [256, 512]
    current_input = None
    
    for i, res in enumerate(resolutions):
        logger.info(f"--- Processing Resolution: {res}x{res} ---")
        
        # Load images at current resolution
        content_img_res = image_loader(content_path, res)
        style_img_res = image_loader(style_path, res)
        
        # Initialize Input
        if current_input is None:
             # First pass: use content as init
             input_img = content_img_res.clone().detach().requires_grad_(True)
        else:
             # Upsample previous result
             with torch.no_grad():
                 input_img = F.interpolate(current_input, size=(res, res), mode='bilinear', align_corners=False)
                 input_img = input_img.detach().requires_grad_(True)
        
        # Get edge mask for this resolution for TV
        edge_mask_res = get_edge_mask(content_img_res)
        tv_module = EdgeAwareTVLoss(1e-4, edge_mask_res)

        # Build Model for this resolution
        model, style_losses, content_losses = get_style_model_and_losses(
            cnn, norm_mean, norm_std, style_img_res, content_img_res, style_config, content_config
        )
        
        # Phase 1: Structure (Aggressive Style, Lower Content, Higher TV)
        # Low Res => Structure lock
        steps = 200 if res == 256 else 300
        
        # Weights
        # Higher resolution needs less TV usually? Or same.
        # We start with strong TV to kill noise, end with weak.
        
        run_optimization(
            model, style_losses, content_losses, tv_module,
            input_img, num_steps=steps,
            style_weight=1e5, content_weight=1e3, # 100k Style, 1k Content
            tv_start=1e-3, tv_end=1e-5,
            phase_name=f"Res-{res}"
        )
        
        # Save for next iteration
        current_input = input_img.detach()

    logger.info("Saving stylized texture...")
    
    # 7. Harden the Pipeline with Sanity Checks
    with torch.no_grad():
        input_img.clamp_(0, 1) # Ensure Valid Range
        
        # Check variance
        var = torch.var(input_img).item() 
        mean = torch.mean(input_img).item()
        
        logger.info(f"Final Image Stats - Mean: {mean:.4f}, Variance: {var:.4f}")
        
        if var < 0.001:
            logger.error("Generated image has extremely low variance (flat color). Style Transfer likely failed.")
            raise RuntimeError(f"Style transfer failed to generate detailed texture (Variance: {var:.4f})")
            
        # Check for channel collapse (grayscale check)
        # simplistic: std dev of (R-G), (G-B)?
        r, g, b = input_img[0, 0, ...], input_img[0, 1, ...], input_img[0, 2, ...]
        color_diff = torch.mean(torch.abs(r - g) + torch.abs(g - b)).item()
        logger.info(f"Color channel divergence: {color_diff:.4f}")
        
    save_image(input_img, output_path)
    logger.info("Style Transfer Completed.")

if __name__ == "__main__":
    pass
