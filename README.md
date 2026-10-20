# 🛡️ AI Model Regression & Load Testing Pipeline

An automated CI/CD testing framework designed to ensure AI models are not only accurate but also reliable, fast, and memory-efficient before they reach production.

## 📖 Overview
Deploying an AI model is easy; ensuring it doesn't break under load is hard. This project provides an end-to-end testing pipeline for PyTorch/FastAPI AI applications. It automatically runs on every GitHub push to validate model accuracy, enforce strict latency SLAs (Service Level Agreements), and simulate high-concurrency traffic to find API bottlenecks.

## ✨ Key Features
*   **Accuracy Regression Testing:** Automatically verifies the model maintains >85% accuracy on a held-out test set, preventing silent model degradation.
*   **Latency SLA Enforcement:** Fails the CI/CD build if single-image inference exceeds **50ms**.
*   **Edge-Case & Shape Validation:** Tests the model against malformed inputs (e.g., NaN values, wrong tensor shapes) to ensure the API fails gracefully without crashing the server.
*   **High-Concurrency Load Testing:** Uses `Locust` to simulate 100+ concurrent users hitting the FastAPI endpoint, measuring P99 latency and throughput.
*   **Automated GitHub Actions:** Runs the entire suite on every `git push`.

## 🛠️ Tech Stack
*   **Machine Learning:** PyTorch, Torchvision, TensorRT
*   **Backend API:** FastAPI, Uvicorn
*   **Testing Framework:** `pytest`, `httpx` (for API testing)
*   **Load Testing:** `Locust`
*   **CI/CD:** GitHub Actions

## 📊 Test Metrics (Results)
*(Note: Update these with your actual test results once you run the pipeline)*
*   **Model Baseline Accuracy:** 87.3%
*   **Single-Image Inference (CPU):** 42ms
*   **Single-Image Inference (TensorRT/GPU):** 8ms
*   **P99 Latency (100 Concurrent Users):** 120ms
*   **API Error Rate under load:** 0.0%

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/darshangr76/ai-testing-pipeline.git
cd ai-testing-pipeline
