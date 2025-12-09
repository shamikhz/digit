# Draw & Guess 🎨🤖

An interactive web application where you draw digits (0-9) and an AI model learns to recognize them better over time!

![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Powered-orange) ![AI Learning](https://img.shields.io/badge/AI-Learns%20From%20You-green) ![License](https://img.shields.io/badge/License-MIT-blue)

## Features

- ✏️ **Interactive Canvas** - Draw digits with mouse or touch
- 🧠 **AI Recognition** - Real-time digit prediction using CNN
- 📚 **Online Learning** - AI learns from your drawings!
- ✅ **Feedback System** - Correct wrong predictions to improve AI
- 📊 **Learning Stats** - Track training samples and accuracy
- 📱 **Responsive Design** - Works on desktop and mobile
- 🌙 **Dark Mode** - Premium glassmorphism UI

## How the AI Learns

1. **Draw a digit** on the canvas
2. **AI predicts** what you drew
3. **Provide feedback**:
   - Click "Correct!" if AI was right → reinforces learning
   - Click the correct digit if AI was wrong → corrects the model
4. **AI improves** with each feedback sample!

The model trains in real-time using your drawings, getting better at recognizing *your* handwriting style.

## Quick Start

1. **Clone or download** this project

2. **Start a local server:**
   ```bash
   python -m http.server 8000
   ```

3. **Open in browser:**
   ```
   http://localhost:8000
   ```

4. **Draw, teach, and watch AI learn!**

## Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript
- **AI/ML:** TensorFlow.js (CNN model)
- **Learning:** Online training with user feedback
- **Storage:** LocalStorage for training stats

## Project Structure

```
digit/
├── index.html      # Main HTML with feedback UI
├── style.css       # Premium dark mode styling
├── script.js       # Canvas drawing + AI learning
└── README.md       # This file
```

## Tips for Best Results

- Draw digits **large and centered**
- Use **smooth, clear strokes**
- Train with **multiple examples** of each digit
- The more you teach, the smarter it gets!

## License

MIT License - feel free to use and modify!
