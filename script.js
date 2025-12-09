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
let trainingData = [];
let totalSamples = 0;
let correctPredictions = 0;

// ===========================
// Initialize Application
// ===========================
document.addEventListener('DOMContentLoaded', async () => {
    initCanvas();
    await createTrainableModel();
    setupEventListeners();
    initPredictionDisplay();
    loadTrainingStats();
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
        saveTrainingStats();
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
        saveTrainingStats();
        updateStatsDisplay();

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
// Training Stats
// ===========================
function saveTrainingStats() {
    localStorage.setItem('drawGuess_totalSamples', totalSamples);
    localStorage.setItem('drawGuess_correctPredictions', correctPredictions);
}

function loadTrainingStats() {
    totalSamples = parseInt(localStorage.getItem('drawGuess_totalSamples') || '0');
    correctPredictions = parseInt(localStorage.getItem('drawGuess_correctPredictions') || '0');
    updateStatsDisplay();
}

function updateStatsDisplay() {
    document.getElementById('sampleCount').textContent = totalSamples;

    const accuracy = totalSamples > 0
        ? ((correctPredictions / totalSamples) * 100).toFixed(1) + '%'
        : '--';
    document.getElementById('accuracyDisplay').textContent = accuracy;
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
