# WheatIQ 🌾

🏆 This project was selected as one of the top final projects of the Data Science & Machine Learning Bootcamp.

An AI-powered bilingual wheat disease assistant for image-based disease detection and interactive guidance.


The system combines deep learning image classification with a conversational chatbot interface to help users identify wheat diseases and receive practical information about symptoms, spread conditions, prevention, and treatment guidance.

The application supports both English and German.

---

# Features

- Wheat vs Non-Wheat image detection
- Healthy vs Diseased wheat classification
- Multi-class wheat disease classification
- Interactive AI chatbot
- English / German bilingual interface
- Disease knowledge base integration
- Confidence-aware predictions
- Gradio-based web interface
- Groq LLM integration
- Internal fallback knowledge system
- Session-based chat memory

---

# Project Architecture

The system uses a multi-stage AI pipeline:

## Stage 1 — Wheat Detection
Determines whether the uploaded image contains wheat or not.

## Stage 2 — Healthy vs Diseased Classification
If wheat is detected, the model predicts whether the plant is healthy or diseased.

## Stage 3 — Disease Classification
If diseased, the system predicts the disease category using a multi-class CNN model.

## Stage 4 — Conversational Assistant
A bilingual chatbot explains:
- disease overview
- causes
- spread mechanisms
- seasonal conditions
- prevention
- treatment guidance

using both:
- internal knowledge base
- Groq LLM

---

# Supported Diseases

Examples include:

- Aphid
- Blast
- Brown Rust
- Yellow Rust
- Black Rust
- Fusarium Head Blight
- Septoria
- Smut
- Tan Spot
- Leaf Blight
- Mildew
- Stem Fly
- Common Root Rot
- Mite
- Healthy Wheat

---

# Tech Stack

## Machine Learning
- TensorFlow / Keras
- EfficientNet
- NumPy
- scikit-learn

## Backend
- Python
- FastAPI

## Frontend
- Gradio

## LLM Integration
- Groq API
- Llama models

## Utilities
- Pillow
- SQLite
- JSON knowledge base

---

# Model Pipeline

## Wheat Detector
Binary classifier:
- wheat
- not wheat

## Binary Health Classifier
Binary classifier:
- healthy
- diseased

## Disease Classifier
Multi-class classifier for disease prediction.

---

# Folder Structure

```text
WheatIQ/
│
├── app/
│   ├── chatbot.py
│   ├── config.py
│   ├── inference.py
│   └── schemas.py
│
├── data/
│   ├── raw_dataset/
│   ├── processed/
│   ├── sample_images/
│   ├── binary_dataset/
│   ├── TEST/
│   └── wheat_detector_dataset/
│
├── gradio_app/
│   ├── gradio_main.py
│   ├── gradio_memory.db
│   ├── logo.png
│   └── requirements.txt
│
├── knowledge_base/
│   ├── diseases.json
│   └── translations.json
│
├── models/
│   ├── wheat_detector/
│   ├── binary/
│   └── keras/
│
├── training/
│   ├── build_binary_dataset.py
│   ├── evaluate_per_class.py
│   ├── evaluate_saved_MultiClassModel.py
│   ├── evaluate_saved_Stage1_DetectorModel.py
│   ├── evaluate_saved_Stage2_BinaryModel.py
│   ├── keras_train.py
│   ├── test_keras_model.py
│   ├── test_wheat_detector.py
│   ├── testCode.py
│   ├── train_binary_classifier.py
│   └── wheat_detector_train.py
│
├── presentation/
│   ├── project_presentation.pdf
│   └── presentation_demo/
│
├── requirements.txt
└── README.md
