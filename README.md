# 🥗 AI-Powered Dietary Guidance for Diverticulitis (Ollama + VLM + MedGemma + VLM)

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

## 1. **Install Dependencies**

You need:
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- [Python 3.11+](https://www.python.org)
- [NVIDIA GPU drivers & CUDA](https://docs.nvidia.com/cuda/)

<img width="927" height="442" alt="image" src="https://github.com/user-attachments/assets/2cb8e2f0-c99f-47a3-9b6d-3305bd96ad21" />
<img width="1298" height="31" alt="image" src="https://github.com/user-attachments/assets/5c9dc425-01f6-45ff-8250-a0ae26e4a6a9" />


- Python packages:  
  ```bash
  pip install gradio ollama pillow
  ```

## 2. **Pull Required Models**

Download these models for Ollama:

```bash
ollama pull gemma3:12b
ollama pull hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M
```

> **Note:** Use smaller VLMs if your GPU VRAM is limited (replace `gemma3:12b` with `gemma3:4b` or similar).

## 3. **Configure for GPU**

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

## 4. **Run the App**

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

[Start]
   |
   v
[Upload Meal Image]
   |
   v
[Enter Digestive Condition]
   |
   v
[Detect Edible Items (VLM Inference)]
   |
   v
[VLM Raw Output]
   |
   v
[User Reviews/Corrects Item List]
   |
   v
[Final Food/Drink List]
   |
   v
[Get Dietary Advice (MedGemma Inference)]
   |
   v
[Show Dietary Guidance Output]
   |
   v
[End]

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

Copyright (c) 2024 Partha Pratim Ray (parthapratimray1986@gmail.com)

Non-Commercial License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, and share the Software for **personal, academic, or research purposes only**,
subject to the following conditions:

1. **Non-Commercial Use Only**:  
   Commercial use, including but not limited to selling, sublicensing, or use in a paid service or product, is strictly prohibited without prior written permission from the copyright holder (Partha Pratim Ray, parthapratimray1986@gmail.com).

2. **Attribution**:  
   You must give appropriate credit to the author (Partha Pratim Ray).

3. **No Derivatives for Commercial Use**:  
   Modifications or derivative works for commercial purposes are strictly prohibited without explicit permission.

4. **Permission for Commercial Use**:  
   If you wish to use this code, concept, or any part thereof for commercial purposes, you must obtain explicit written permission from the copyright holder (Partha Pratim Ray).  
   Unauthorized commercial use is a violation of this license and will be considered an offense.

5. **Disclaimer**:  
   The software is provided "as is", without warranty of any kind, express or implied.

For inquiries about commercial use or licensing, contact:  
Partha Pratim Ray  
parthapratimray1986@gmail.com

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
<img width="1853" height="883" alt="Screenshot from 2025-07-19 20-30-33" src="https://github.com/user-attachments/assets/f9d65936-3380-4ed6-84ed-70d96410b2bc" />
<img width="1849" height="801" alt="Screenshot from 2025-07-19 20-31-20" src="https://github.com/user-attachments/assets/c96bd188-08e7-4a89-b1ad-8c3350e2a1bf" />



---

