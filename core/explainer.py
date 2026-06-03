import torch
import clip
from PIL import Image
import numpy as np
import cv2

class XAIExplainer:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def generate_heatmap(self, image_path, text_label, save_path):
        """
        Generates an attention heatmap based on feature activation magnitude.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 1. Get the raw visual tokens from the first convolution layer
                # Shape is usually [1, 768, 7, 7] for ViT-B/32 (7x7 grid = 49 patches)
                x = self.model.visual.conv1(image_input.type(self.model.dtype))
                
                # 2. Flatten the grid
                x = x.reshape(x.shape[0], x.shape[1], -1) # Shape: [1, 768, 49]
                x = x.permute(0, 2, 1)  # Shape: [1, 49, 768]
                
                # 3. Calculate activation magnitude (Norm)
                activation = x.norm(dim=2, keepdim=True) # Shape: [1, 49, 1]
                
                # 4. Reshape back to grid
                # FIX: We use the actual square root of the patches (sqrt(49) = 7)
                num_patches = activation.shape[1]
                grid_size = int(np.sqrt(num_patches))
                
                heatmap = activation.reshape(1, grid_size, grid_size)
                heatmap = heatmap[0].detach().cpu().numpy()
                
            # 5. Resize heatmap to original image size
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            
            # 6. Normalize to 0-255
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            heatmap = np.uint8(255 * heatmap)

            # 7. Apply Color Map (Jet)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 8. Overlay on original image
            original_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            superimposed = cv2.addWeighted(heatmap_color, 0.5, original_cv, 0.5, 0)
            
            cv2.imwrite(save_path, superimposed)
            return save_path

        except Exception as e:
            print(f"[Explainer Error] {e}")
            raise e