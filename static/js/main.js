const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');

// Panels
const uploadPanel = document.getElementById('upload-panel');
const resultsPanel = document.getElementById('results-panel');

// Result Elements
const resultImg = document.getElementById('result-img');
const heatmapOverlay = document.getElementById('heatmap-overlay');
const predictionsList = document.getElementById('predictions-list');
const topLabelDisplay = document.getElementById('top-label');
const toggleHeatmapBtn = document.getElementById('toggle-heatmap');

// --- Drag & Drop Logic ---
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

function handleFile(file) {
    if (file) {
        dropZone.innerHTML = `
            <div class="icon-container" style="border-color: #bc13fe;">
               <h3 style="color: #bc13fe; margin:0;">FILE SELECTED</h3>
            </div>
            <h3 style="margin-top:10px;">${file.name}</h3>
        `;
    }
}

// --- ANALYSIS LOGIC ---
analyzeBtn.addEventListener('click', () => {
    if (!fileInput.files[0]) {
        alert('Please upload an image first!');
        return;
    }

    // UI Loading State
    statusBar.classList.remove('hidden');
    statusText.innerText = "Initializing Neural Core...";
    analyzeBtn.disabled = true;

    // Simulate tech steps
    setTimeout(() => { statusText.innerText = "Enhancing Resolution (ESRGAN)..."; }, 1000);
    setTimeout(() => { statusText.innerText = "Running Zero-Shot Inference..."; }, 3000);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('enhancer', document.getElementById('model-select').value);
    formData.append('mode', document.getElementById('analysis-mode').value);
    
    // NEW: Capture custom labels
    formData.append('custom_labels', document.getElementById('custom-labels').value);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if(data.error) {
            alert("Error: " + data.error);
            location.reload();
            return;
        }

        // --- SUCCESS: SWITCH PANELS ---
        uploadPanel.classList.add('hidden');
        resultsPanel.classList.remove('hidden');

        // 1. Set Image
        resultImg.src = data.processed_image;

        // 2. Set Heatmap (if available)
        if (data.heatmap_url) {
            heatmapOverlay.src = data.heatmap_url;
            toggleHeatmapBtn.classList.remove('disabled');
            toggleHeatmapBtn.onclick = () => {
                heatmapOverlay.classList.toggle('hidden');
                toggleHeatmapBtn.innerText = heatmapOverlay.classList.contains('hidden') 
                    ? "SHOW X-RAY (HEATMAP)" 
                    : "HIDE X-RAY";
            };
        } else {
            toggleHeatmapBtn.classList.add('disabled');
            toggleHeatmapBtn.innerText = "X-RAY UNAVAILABLE";
        }

        // 3. Render Predictions
        renderPredictions(data.predictions);

        // 4. Set Top Label
        topLabelDisplay.innerText = data.predictions[0].label;
    })
    .catch(error => {
        console.error('Error:', error);
        statusText.innerText = "System Error.";
        alert("Something went wrong. Check console.");
    });
});

function renderPredictions(predictions) {
    predictionsList.innerHTML = '';
    // Take top 5
    predictions.slice(0, 5).forEach(pred => {
        const score = pred.score.toFixed(2);
        const item = document.createElement('div');
        item.className = 'pred-item';
        item.innerHTML = `
            <div class="pred-label">
                <span>${pred.label}</span>
                <span>${score}%</span>
            </div>
            <div class="progress-bg">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
        `;
        predictionsList.appendChild(item);
        
        // Animate bar after small delay
        setTimeout(() => {
            item.querySelector('.progress-fill').style.width = `${score}%`;
        }, 100);
    });
}