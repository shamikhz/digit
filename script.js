// ===========================
// Firebase Configuration
// ===========================
const firebaseConfig = {
    apiKey: "AIzaSyD0XTsjeH8CKKIkVD5U5KKKkaTUf92hB-w",
    authDomain: "harmonia-dmbqa.firebaseapp.com",
    projectId: "harmonia-dmbqa",
    storageBucket: "harmonia-dmbqa.firebasestorage.app",
    messagingSenderId: "985369858331",
    appId: "1:985369858331:web:916406ca3d10f11d0003be"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();
const storage = firebase.storage();

// ===========================
// Global Variables
// ===========================
let canvas, ctx;
let isDrawing = false;
let model = null;
let lastX = 0;
let lastY = 0;
let currentImageTensor = null;
let currentPrediction = null;
let totalSamples = 0;
let correctPredictions = 0;

// Unique session ID for this user/browser
const SESSION_ID = getOrCreateSessionId();

function getOrCreateSessionId() {
    let id = localStorage.getItem('drawGuess_sessionId');
    if (!id) {
        id = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('drawGuess_sessionId', id);
    }
    return id;
}

// ===========================
// Initialize Application
// ===========================
document.addEventListener('DOMContentLoaded', async () => {
    initCanvas();
    await createTrainableModel();
    setupEventListeners();
    initPredictionDisplay();
    await loadFromFirebase();
    hideFeedbackSection();
});

// ===========================
// Canvas Initialization
// ===========================
function initCanvas() {
    canvas = document.getElementById('drawCanvas');
    ctx = canvas.getContext('2d');

    // Fill canvas with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set up drawing styling
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';
}

// ===========================
// Create Trainable Model
// ===========================
async function createTrainableModel() {
    updateStatus('Creating AI model...', 'loading');
    console.log('Creating trainable MNIST model...');

    // Create a CNN model for MNIST
    model = tf.sequential();

    // First convolutional layer
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // Second convolutional layer
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // Flatten and dense layers
    model.add(tf.layers.flatten());
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    // Compile for training
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    console.log('Trainable model created!');
    updateStatus('Ready - Draw to start!', 'ready');
}

// ===========================
// Event Listeners Setup
// ===========================
function setupEventListeners() {
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', stopDrawing, { passive: false });

    // Clear button
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);

    // Digit buttons for correction
    document.querySelectorAll('.digit-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const digit = parseInt(e.target.dataset.digit);
            trainOnDigit(digit);
        });
    });

    // Correct button
    document.getElementById('correctBtn').addEventListener('click', () => {
        if (currentPrediction !== null) {
            trainOnDigit(currentPrediction, true);
        }
    });

    // Save model button
    document.getElementById('saveModelBtn').addEventListener('click', saveModelToFirebase);

    // Reset model button
    document.getElementById('resetModelBtn').addEventListener('click', resetModel);
}

// ===========================
// Drawing Functions
// ===========================
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    lastX = (e.clientX - rect.left) * scaleX;
    lastY = (e.clientY - rect.top) * scaleY;

    // Draw a dot at the start
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();

    // Hide overlay hint
    document.getElementById('canvasOverlay').classList.add('hidden');
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX = x;
    lastY = y;
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;

    // Predict after drawing stops
    setTimeout(() => {
        predict();
    }, 300);
}

// Touch event handlers
function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    lastX = (touch.clientX - rect.left) * scaleX;
    lastY = (touch.clientY - rect.top) * scaleY;
    isDrawing = true;

    // Draw a dot at the start
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();

    document.getElementById('canvasOverlay').classList.add('hidden');
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (touch.clientX - rect.left) * scaleX;
    const y = (touch.clientY - rect.top) * scaleY;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX = x;
    lastY = y;
}

// ===========================
// Clear Canvas
// ===========================
function clearCanvas() {
    // Clear and fill with white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Reset stroke style
    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'black';

    // Clean up current image tensor
    if (currentImageTensor) {
        currentImageTensor.dispose();
        currentImageTensor = null;
    }
    currentPrediction = null;

    // Reset prediction display
    document.getElementById('mainPrediction').innerHTML = `
        <div class="predicted-digit">?</div>
        <div class="confidence-label">Draw to predict</div>
    `;

    // Clear all predictions
    initPredictionDisplay();

    // Hide feedback section
    hideFeedbackSection();

    // Show overlay hint
    document.getElementById('canvasOverlay').classList.remove('hidden');
}

// ===========================
// Prediction Functions
// ===========================
async function predict() {
    if (!model) {
        console.error('Model not loaded yet');
        updateStatus('Model not ready', 'error');
        return;
    }

    // Check if canvas is empty
    if (isCanvasEmpty()) {
        console.log('Canvas is empty, skipping prediction');
        return;
    }

    updateStatus('Analyzing...', 'analyzing');

    try {
        // Clean up previous tensor
        if (currentImageTensor) {
            currentImageTensor.dispose();
        }

        // Preprocess the canvas image
        currentImageTensor = preprocessCanvas();

        // Make prediction
        const prediction = model.predict(currentImageTensor);
        const predictionData = await prediction.data();
        prediction.dispose();

        console.log('Prediction:', predictionData);

        // Find best prediction
        let maxIndex = 0;
        let maxConfidence = predictionData[0];
        for (let i = 1; i < predictionData.length; i++) {
            if (predictionData[i] > maxConfidence) {
                maxConfidence = predictionData[i];
                maxIndex = i;
            }
        }
        currentPrediction = maxIndex;

        // Display results
        displayPrediction(predictionData);

        // Show feedback section
        showFeedbackSection();

        updateStatus('Ready', 'ready');
    } catch (error) {
        console.error('Prediction error:', error);
        updateStatus('Prediction failed', 'error');
    }
}

// ===========================
// Image Preprocessing
// ===========================
function preprocessCanvas() {
    // Get image data from canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Convert to grayscale tensor
    let tensor = tf.browser.fromPixels(imageData, 1);

    // Resize to 28x28 (MNIST input size)
    const resized = tf.image.resizeBilinear(tensor, [28, 28]);
    tensor.dispose();

    // Invert colors (MNIST expects white digits on black background)
    const inverted = tf.scalar(255).sub(resized);
    resized.dispose();

    // Normalize to [0, 1]
    const normalized = inverted.div(tf.scalar(255));
    inverted.dispose();

    // Reshape for model input [1, 28, 28, 1]
    const batched = normalized.expandDims(0);
    normalized.dispose();

    return batched;
}

// ===========================
// Check if Canvas is Empty
// ===========================
function isCanvasEmpty() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
        if (data[i] < 250 || data[i + 1] < 250 || data[i + 2] < 250) {
            return false;
        }
    }
    return true;
}

// ===========================
// Train on Digit (Online Learning)
// ===========================
async function trainOnDigit(digit, isCorrect = false) {
    if (!currentImageTensor || !model) {
        console.error('No image or model available for training');
        return;
    }

    updateStatus('Learning...', 'analyzing');

    try {
        // Create one-hot encoded label
        const label = tf.oneHot(tf.tensor1d([digit], 'int32'), 10);

        // Clone the current image tensor for training
        const trainImage = currentImageTensor.clone();

        // Train the model on this single sample
        await model.fit(trainImage, label, {
            epochs: 3,
            batchSize: 1,
            verbose: 0
        });

        // Clean up
        label.dispose();
        trainImage.dispose();

        // Update stats
        totalSamples++;
        if (isCorrect) {
            correctPredictions++;
        }

        // Save to Firebase
        await saveStatsToFirebase();
        updateStatsDisplay();

        // Show success message
        const message = isCorrect
            ? `✓ Learned! AI confirmed digit ${digit}`
            : `✓ Corrected! AI learned digit ${digit}`;

        updateStatus(message, 'ready');

        // Flash the feedback section
        document.getElementById('feedbackSection').classList.add('trained');
        setTimeout(() => {
            document.getElementById('feedbackSection').classList.remove('trained');
        }, 500);

        console.log(`Trained on digit ${digit}. Total samples: ${totalSamples}`);

        // Re-predict to show updated model
        setTimeout(async () => {
            await predict();
        }, 500);

    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Training failed', 'error');
    }
}

// ===========================
// Firebase: Save Stats
// ===========================
async function saveStatsToFirebase() {
    try {
        await db.collection('drawGuess').doc(SESSION_ID).set({
            totalSamples: totalSamples,
            correctPredictions: correctPredictions,
            lastUpdated: firebase.firestore.FieldValue.serverTimestamp()
        }, { merge: true });

        updateSyncStatus('✅ Synced');
        console.log('Stats saved to Firebase');
    } catch (error) {
        console.error('Error saving stats to Firebase:', error);
        updateSyncStatus('❌ Sync failed');
    }
}

// ===========================
// Firebase: Load Stats and Model
// ===========================
async function loadFromFirebase() {
    updateSyncStatus('🔄 Loading...');

    try {
        // Load stats from Firestore
        const doc = await db.collection('drawGuess').doc(SESSION_ID).get();

        if (doc.exists) {
            const data = doc.data();
            totalSamples = data.totalSamples || 0;
            correctPredictions = data.correctPredictions || 0;
            console.log('Stats loaded from Firebase:', data);
        }

        // Try to load model weights from Storage
        await loadModelFromFirebase();

        updateStatsDisplay();
        updateSyncStatus('✅ Connected');

    } catch (error) {
        console.error('Error loading from Firebase:', error);
        updateSyncStatus('❌ Connection failed');
    }
}

// ===========================
// Firebase: Save Model Weights
// ===========================
async function saveModelToFirebase() {
    if (!model) {
        alert('No model to save!');
        return;
    }

    updateStatus('Saving model to cloud...', 'analyzing');
    updateSyncStatus('🔄 Uploading...');

    try {
        // Save model to a temporary location using TensorFlow.js
        const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
            // Convert model weights to JSON string
            const weightsData = {
                modelTopology: artifacts.modelTopology,
                weightSpecs: artifacts.weightSpecs,
                weightData: Array.from(new Uint8Array(artifacts.weightData))
            };

            // Save to Firestore (for smaller models) or Storage
            const modelJson = JSON.stringify(weightsData);
            const blob = new Blob([modelJson], { type: 'application/json' });

            // Upload to Firebase Storage
            const storageRef = storage.ref();
            const modelRef = storageRef.child(`models/${SESSION_ID}/model.json`);
            await modelRef.put(blob);

            console.log('Model saved to Firebase Storage');
            return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
        }));

        updateStatus('Model saved to cloud!', 'ready');
        updateSyncStatus('✅ Model saved');

        // Also update the stats with model save timestamp
        await db.collection('drawGuess').doc(SESSION_ID).set({
            modelSaved: firebase.firestore.FieldValue.serverTimestamp()
        }, { merge: true });

    } catch (error) {
        console.error('Error saving model to Firebase:', error);
        updateStatus('Failed to save model', 'error');
        updateSyncStatus('❌ Save failed');
    }
}

// ===========================
// Firebase: Load Model Weights
// ===========================
async function loadModelFromFirebase() {
    try {
        const storageRef = storage.ref();
        const modelRef = storageRef.child(`models/${SESSION_ID}/model.json`);

        // Check if model exists
        const url = await modelRef.getDownloadURL();

        // Download model data
        const response = await fetch(url);
        const weightsData = await response.json();

        // Reconstruct the weights
        const weightData = new Uint8Array(weightsData.weightData).buffer;

        // Load weights into model
        const weightSpecs = weightsData.weightSpecs;
        const weights = tf.io.decodeWeights(weightData, weightSpecs);

        // Set weights to model
        let weightIndex = 0;
        for (const layer of model.layers) {
            const layerWeights = layer.getWeights();
            if (layerWeights.length > 0) {
                const newWeights = [];
                for (let i = 0; i < layerWeights.length; i++) {
                    const weightName = Object.keys(weights)[weightIndex];
                    if (weightName && weights[weightName]) {
                        newWeights.push(weights[weightName]);
                        weightIndex++;
                    }
                }
                if (newWeights.length === layerWeights.length) {
                    layer.setWeights(newWeights);
                }
            }
        }

        console.log('Model loaded from Firebase Storage');
        updateStatus('Model loaded from cloud!', 'ready');

    } catch (error) {
        // Model doesn't exist yet - that's okay
        if (error.code === 'storage/object-not-found') {
            console.log('No saved model found - using fresh model');
        } else {
            console.error('Error loading model from Firebase:', error);
        }
    }
}

// ===========================
// Reset Model
// ===========================
async function resetModel() {
    if (confirm('Reset the AI model? All learning will be lost.')) {
        updateStatus('Resetting model...', 'loading');

        // Dispose old model
        if (model) {
            model.dispose();
        }

        // Create new model
        await createTrainableModel();

        // Reset stats
        totalSamples = 0;
        correctPredictions = 0;
        await saveStatsToFirebase();
        updateStatsDisplay();

        // Delete saved model from Firebase
        try {
            const storageRef = storage.ref();
            const modelRef = storageRef.child(`models/${SESSION_ID}/model.json`);
            await modelRef.delete();
        } catch (e) {
            // Model might not exist
        }

        // Clear canvas
        clearCanvas();

        updateStatus('Model reset! Ready to learn.', 'ready');
    }
}

// ===========================
// Feedback Section Visibility
// ===========================
function showFeedbackSection() {
    document.getElementById('feedbackSection').style.display = 'block';
}

function hideFeedbackSection() {
    document.getElementById('feedbackSection').style.display = 'none';
}

// ===========================
// Update Stats Display
// ===========================
function updateStatsDisplay() {
    document.getElementById('sampleCount').textContent = totalSamples;

    const accuracy = totalSamples > 0
        ? ((correctPredictions / totalSamples) * 100).toFixed(1) + '%'
        : '--';
    document.getElementById('accuracyDisplay').textContent = accuracy;
}

// ===========================
// Update Sync Status
// ===========================
function updateSyncStatus(status) {
    const syncEl = document.getElementById('syncStatus');
    if (syncEl) {
        syncEl.textContent = status;
    }
}

// ===========================
// Display Prediction Results
// ===========================
function displayPrediction(predictions) {
    const predArray = Array.from(predictions);

    let maxIndex = 0;
    let maxConfidence = predArray[0];

    for (let i = 1; i < predArray.length; i++) {
        if (predArray[i] > maxConfidence) {
            maxConfidence = predArray[i];
            maxIndex = i;
        }
    }

    console.log(`Predicted digit: ${maxIndex} with confidence: ${(maxConfidence * 100).toFixed(1)}%`);

    const mainPrediction = document.getElementById('mainPrediction');
    mainPrediction.innerHTML = `
        <div class="predicted-digit">${maxIndex}</div>
        <div class="confidence-label">Confidence</div>
        <div class="confidence-value">${(maxConfidence * 100).toFixed(1)}%</div>
    `;

    updateAllPredictions(predArray);
}

// ===========================
// Initialize Prediction Display
// ===========================
function initPredictionDisplay() {
    const predictionsGrid = document.querySelector('.predictions-grid');
    if (!predictionsGrid) return;

    predictionsGrid.innerHTML = '';

    for (let i = 0; i < 10; i++) {
        const item = createPredictionItem(i, 0);
        predictionsGrid.appendChild(item);
    }
}

// ===========================
// Update All Predictions
// ===========================
function updateAllPredictions(predictions) {
    const predictionsGrid = document.querySelector('.predictions-grid');
    if (!predictionsGrid) return;

    predictionsGrid.innerHTML = '';

    const sortedPredictions = Array.from(predictions)
        .map((confidence, digit) => ({ digit, confidence }))
        .sort((a, b) => b.confidence - a.confidence);

    sortedPredictions.forEach((pred, index) => {
        const item = createPredictionItem(pred.digit, pred.confidence);
        item.style.animationDelay = `${index * 0.05}s`;
        predictionsGrid.appendChild(item);
    });
}

// ===========================
// Create Prediction Item
// ===========================
function createPredictionItem(digit, confidence) {
    const item = document.createElement('div');
    item.className = 'prediction-item';

    const percentage = (confidence * 100).toFixed(1);

    item.innerHTML = `
        <div class="prediction-digit">${digit}</div>
        <div class="prediction-bar-container">
            <div class="prediction-bar-bg">
                <div class="prediction-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="prediction-percentage">${percentage}%</div>
        </div>
    `;

    return item;
}

// ===========================
// Update Status Indicator
// ===========================
function updateStatus(text, status) {
    const statusIndicator = document.getElementById('statusIndicator');
    if (!statusIndicator) return;

    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');

    if (statusText) statusText.textContent = text;

    if (statusDot) {
        switch (status) {
            case 'ready':
                statusDot.style.background = 'var(--success)';
                break;
            case 'loading':
            case 'analyzing':
                statusDot.style.background = 'var(--accent-primary)';
                break;
            case 'error':
                statusDot.style.background = 'var(--danger)';
                break;
        }
    }
}
