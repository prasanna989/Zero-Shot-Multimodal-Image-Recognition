import os
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# --- IMPORTING THE BRAIN (Your Core Modules) ---
from core.enhancer import ImageEnhancer
from core.recognizer import ZeroShotRecognizer
from core.explainer import XAIExplainer

# Initialize Flask App
app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

# --- MODEL INITIALIZATION (Loads once at startup) ---
print(">>> SYSTEM STARTUP: Loading AI Models... This may take a moment.")

try:
    # 1. Load Real-ESRGAN (Super Resolution)
    enhancer = ImageEnhancer(model_name='RealESRGAN_x4plus')
    
    # 2. Load CLIP (Zero-Shot Recognition)
    recognizer = ZeroShotRecognizer(model_name="ViT-B/32")
    
    # 3. Load XAI Explainer (Heatmaps)
    explainer = XAIExplainer(model_name="ViT-B/32")
    
    print(">>> SYSTEM READY: All Neural Cores Online.")
except Exception as e:
    print(f">>> CRITICAL ERROR: Could not load models. {e}")
    pass


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main Logic Pipeline:
    1. Save Image
    2. Enhance (Optional)
    3. Recognize (CLIP)
    4. Explain (Optional XAI)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 1. SAVE THE RAW IMAGE
        filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Get User Options
        use_enhancement = request.form.get('enhancer') 
        analysis_mode = request.form.get('mode')       
        
        # --- DYNAMIC LABEL LOGIC (THE UPGRADE) ---
        user_labels = request.form.get('custom_labels')
        
        if user_labels and user_labels.strip() != "":
            # Process user input: "cat, dog" -> ["a cat", "a dog"]
            labels = [l.strip() for l in user_labels.split(',') if l.strip()]
            print(f">>> INFO: Using Custom Labels: {labels}")
        else:
            # Fallback to Defaults
            labels = [
                "a person", "a dog", "a cat", "a car", 
                "suspicious activity", "fire", "a weapon", 
                "nature", "urban city", "food",
                "sports ball", "cricket bat", "football", "basketball", "gym equipment"
            ]
            print(">>> INFO: Using Default Label List")

        # --- PIPELINE START ---
        print(f"\n[PIPELINE] Processing file: {filename}")
        
        # STAGE 1: ENHANCEMENT
        current_image_path = upload_path
        processed_image_url = f"static/uploads/{filename}" 
        
        if use_enhancement == 'esrgan':
            print(">>> STATUS: Starting ESRGAN Enhancement (this may take 10-20s)...")
            try:
                enhanced_filename = f"enhanced_{filename}"
                enhanced_path = os.path.join(app.config['RESULT_FOLDER'], enhanced_filename)
                
                result_path = enhancer.enhance(upload_path, enhanced_path)
                
                if result_path:
                    current_image_path = result_path
                    processed_image_url = f"static/results/{enhanced_filename}"
                    print(">>> STATUS: Enhancement Complete.")
            except Exception as e:
                print(f"[ERROR] Enhancement Failed: {e}")

        # STAGE 2: RECOGNITION (Zero-Shot)
        print(">>> STATUS: Running CLIP Recognition...")
        try:
            predictions = recognizer.predict(current_image_path, labels)
            top_label = predictions[0]['label']
            print(f">>> RESULT: Top Prediction = '{top_label}' ({predictions[0]['score']:.2f}%)")
        except Exception as e:
            print(f"[ERROR] Recognition Failed: {e}")
            return jsonify({'error': str(e)}), 500

        # STAGE 3: EXPLANATION (Heatmap)
        heatmap_url = None
        
        if analysis_mode == 'explainable':
            print(f">>> STATUS: Generating Heatmap for '{top_label}'...")
            try:
                heatmap_filename = f"heatmap_{filename}"
                heatmap_path = os.path.join(app.config['RESULT_FOLDER'], heatmap_filename)
                
                explainer.generate_heatmap(current_image_path, top_label, heatmap_path)
                heatmap_url = f"static/results/{heatmap_filename}"
                print(">>> STATUS: Heatmap Generated Successfully.")
            except Exception as e:
                print(f"[ERROR] XAI Failed: {e}")
        else:
             print(">>> STATUS: XAI Skipped (Mode set to Standard).")

        # --- RESPONSE ---
        return jsonify({
            'status': 'success',
            'original_image': f"static/uploads/{filename}",
            'processed_image': processed_image_url,
            'predictions': predictions,
            'heatmap_url': heatmap_url,
            'mode': analysis_mode
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)