import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torch.hub import download_url_to_file

class ImageEnhancer:
    def __init__(self, model_name='RealESRGAN_x4plus'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Enhancer] Loading {model_name} on {self.device}...")
        
        # 1. Define Model URL and Local Path explicitly
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model_path = os.path.join('models', 'RealESRGAN_x4plus.pth')
        
        # Ensure 'models' folder exists
        os.makedirs('models', exist_ok=True)

        # 2. Download weights if missing
        if not os.path.exists(model_path):
            print(f"[Enhancer] Model not found. Downloading to {model_path}...")
            try:
                download_url_to_file(model_url, model_path)
                print("[Enhancer] Download complete.")
            except Exception as e:
                print(f"[Enhancer] Download Failed: {e}")
                raise e

        # 3. Load the model architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # 4. Initialize the Restorer with the EXPLICIT path
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path, # <--- passing the actual file path now
            model=model,
            tile=0, 
            tile_pad=10,
            pre_pad=0,
            half=False, 
            device=self.device,
        )

    def enhance(self, image_path, output_path):
        """
        Reads an image, upscales it 4x, and saves it.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not read image.")

            # Run the model
            output, _ = self.upsampler.enhance(img, outscale=4)
            
            # Save result
            cv2.imwrite(output_path, output)
            return output_path
        except Exception as e:
            print(f"Error in enhancement: {e}")
            return None