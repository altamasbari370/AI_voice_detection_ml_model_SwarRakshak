# 🛡️ SwarRakshak (VoiceShield AI)
**Real-Time AI Audio Deepfake & Voice Clone Detection System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-009688.svg?logo=fastapi&logoColor=white)]()

> **Built for Build with Bharat 2.0:** Protecting Digital India from AI voice cloning, digital arrest scams, and financial impersonation.

---

## 📌 Problem Overview
Generative AI voice models (ElevenLabs, Bark, Tacotron) can clone human voices in seconds, driving a major rise in impersonation fraud and social engineering attacks.

## 💡 Solution
**SwarRakshak** is a GPU-accelerated deep learning microservice that analyzes micro-level acoustic spectral artifacts to detect cloned voices in real time.

### ✨ Key Highlights
* **Low Latency:** In-memory audio processing in `< 50 ms` (0 disk I/O).
* **High Precision:** Accuracy `> 99.2%` with validation loss `0.0260`.
* **Noise Immunity:** Max-Feature-Map (MFM) activation filters out ambient background noise.
* **Production API:** Asynchronous FastAPI backend with OpenAPI / Swagger UI.

---

## 🏗️ System Architecture

```text
[ 🎙️ Audio Input (.wav) ] 
           │
           ▼
[ ⚡ Standardization (16kHz Mono, 2.0s Tail-Aligned) ] 
           │
           ▼
[ 📊 80-Band Log-Mel Spectrogram Extraction ] 
           │
           ▼
[ 🧠 Light-CNN (LCNN) Deep Classifier (MFM Layers) ] 
           │
           ▼
[ 🛡️ API Output: REAL vs. FAKE + Confidence Score ]