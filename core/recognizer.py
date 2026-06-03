import torch
import clip
from PIL import Image

class ZeroShotRecognizer:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Recognizer] Loading CLIP {model_name} on {self.device}...")
        
        # Load CLIP
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def predict(self, image_path, labels):
        """
        Performs Zero-Shot Classification.
        labels: List of strings e.g., ["a cat", "a dog", "a car"]
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            # Calculate similarity
            logits_per_image, _ = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Format results
        results = []
        for i, label in enumerate(labels):
            results.append({
                "label": label,
                "score": float(probs[0][i]) * 100 # Convert to percentage
            })
        
        # Sort by highest confidence
        results.sort(key=lambda x: x['score'], reverse=True)
        return results