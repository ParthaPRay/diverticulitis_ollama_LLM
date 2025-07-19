# 🥗 AI-Powered Dietary Guidance for Diverticulitis (Ollama + VLM + MedGemma)

> **AI assistant for meal analysis and dietary recommendations in Diverticulitis using local Vision-Language Model (VLM) and clinical LLMs – GPU Accelerated!**

---

## ✨ Features

- **Upload a meal photo:** Automatically detects all visible foods and drinks using a powerful VLM (e.g., Gemma3 12B).
- **Automatic food listing:** Extracts all edible items with confidence scores.
- **User correction:** Add missed items before analysis.
- **Medical dietary analysis:** Uses a clinical LLM (MedGemma) to classify each food as *Safe*, *Unsafe*, or *Caution* for patients with Diverticulitis.
- **Practical dietary summary:** Receives meal-specific advice with full rationale.
- **Runs entirely locally on your GPU** (requires NVIDIA and Ollama).
- **Modern Gradio interface** for easy use in browser.

---

## 🚀 Quickstart

### 1. **Install Dependencies**

You need:
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- [Python 3.9+](https://www.python.org)
- [NVIDIA GPU drivers & CUDA](https://docs.nvidia.com/cuda/)
- Python packages:  
  ```bash
  pip install gradio ollama pillow
````

### 2. **Pull Required Models**

Download these models for Ollama:

```bash
ollama pull gemma3:12b
ollama pull hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M
```

> **Note:** Use smaller VLMs if your GPU VRAM is limited (replace `gemma3:12b` with `gemma3:4b` or similar).

### 3. **Configure for GPU**

Ollama uses GPU by default if supported:

* Start Ollama with:

  ```bash
  ollama serve
  ```
* Check GPU status:

  ```bash
  ollama ps
  ```

  Output should show `100% GPU` in the processor column.

If you have multiple GPUs, restrict usage via:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### 4. **Run the App**

```bash
python app.py
```

The Gradio interface will open at [http://localhost:7860](http://localhost:7860).

---

## 🖼️ Example Workflow

1. **Upload your meal image**
2. **Describe your digestive condition** (e.g., "History of diverticulitis, currently asymptomatic")
3. **Click "Detect Edible Items"**
4. **Review and add any missed food items**
5. **Click "Get Dietary Advice"** to see the MedGemma table and summary

---

## ⚡️ GPU & Performance Notes

* This app is designed for **GPU inference**. Models load/unload between tasks to maximize VRAM for multi-stage analysis.
* Check real-time GPU usage with `nvidia-smi`.
* If you observe CPU fallback, ensure:

  * You have correct NVIDIA/CUDA drivers installed.
  * Your model fits in GPU memory (`ollama ps` shows `100% GPU`).

---

## 🛠️ Code Structure

* **app.py**: Main application (see [code](./app.py))
* **VLM (Vision-Language Model)**: Extracts all food/drink items from the image.
* **MedGemma**: Classifies foods for Diverticulitis safety, outputs Markdown table + summary.

---

## 🧑‍💻 Authors & Credits

* **Developed by:** \[Your Name]
* **Powered by:** [Ollama](https://ollama.com), [Gradio](https://gradio.app), [MedGemma LLM](https://huggingface.co/unsloth/medgemma-4b-it-GGUF)

---

## 📝 License

[MIT License](LICENSE)

---

## 🤝 Contributing

Pull requests and suggestions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) if available.

---

## 💬 Troubleshooting

* **Model loading fails?** Check VRAM with `nvidia-smi`, lower model size if needed.
* **Still slow?** Try smaller quantized models, or increase VRAM.
* **Ollama uses CPU?** Ensure `ollama ps` shows `100% GPU` and check your CUDA setup.

---

## 📷 Screenshot

> *Add a screenshot of the app running with an example image and output table!*

---

