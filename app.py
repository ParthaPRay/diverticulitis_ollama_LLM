# =============================================================================
# Title:    AI-powered Dietary Guidance for Diverticulitis Disease (Ollama + Gradio)
# Author:   Partha Pratim Ray
# Email:    parthapratimray1986@gmail.com
# Copyright (c) 2024 Partha Pratim Ray
#
# License Notice:
#   - This code and concept are provided strictly for non-commercial use only.
#   - Commercial use of any kind (including resale, SaaS, for-profit services, or incorporation in a commercial product)
#     requires prior written permission from the author (Partha Pratim Ray).
#   - Unauthorized commercial use is a violation of the license and is considered an offense.
#
#   For permission, please contact: parthapratimray1986@gmail.com
#
# Disclaimer:
#   - This software is provided "as is", without warranty of any kind.
# =============================================================================

import gradio as gr
import requests
import base64
from PIL import Image
from io import BytesIO
import traceback
import time
import csv
import os
from datetime import datetime

# --- Configurable model names ---
VLM_MODEL = "gemma3:12b"
MED_MODEL = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"
OLLAMA_HOST = "http://localhost:11434"

VLM_PROMPT = (
    "Carefully and comprehensively analyze the given image. "
    "List every visible food, fruit, drink, salt, condiment, packaged, prepared, or edible item present on any platter, plate, bowl, tray, glass, cup, or the table surface. "
    "Include all main dishes, sides, breads, grains, vegetables, fruits, salads, sauces, dips, garnishes, snacks, beverages, salts, spices, and any packaged or processed foods or drinks you can visually identify. "
    "Do not list non-edible items such as cup, glass, spoon etc."
    "For each item, provide its exact name and your confidence score (0-100) of correct identification. "
    "Do not omit or merge similar items—list every distinguishable edible item, even if multiple of the same type appear. "
    "Do not describe the scene or background, do not output in JSON, just output a clean, numbered list in this format:\n"
    "1. <item> - <confidence score>\n"
    "2. <item> - <confidence score>\n"
)

def image_to_base64_jpeg(image):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_ollama_metrics(response, is_chat=False):
    # Extract Ollama API metrics for /api/generate or /api/chat
    # is_chat: if using chat API, values are at root, 'response' text is in message['content']
    if is_chat:
        total_duration = response.get('total_duration')
        load_duration = response.get('load_duration')
        prompt_eval_count = response.get('prompt_eval_count')
        prompt_eval_duration = response.get('prompt_eval_duration')
        eval_count = response.get('eval_count')
        eval_duration = response.get('eval_duration')
        tokens_per_second = (eval_count / (eval_duration / 1e9)) if eval_count and eval_duration else None
        response_text = (response.get("message") or {}).get("content", None)
    else:
        total_duration = response.get('total_duration')
        load_duration = response.get('load_duration')
        prompt_eval_count = response.get('prompt_eval_count')
        prompt_eval_duration = response.get('prompt_eval_duration')
        eval_count = response.get('eval_count')
        eval_duration = response.get('eval_duration')
        tokens_per_second = (eval_count / (eval_duration / 1e9)) if eval_count and eval_duration else None
        response_text = response.get('response', None)
    return [
        total_duration, load_duration, prompt_eval_count, prompt_eval_duration,
        eval_count, eval_duration, tokens_per_second, response_text
    ]

def log_metrics_csv(
    vlm_model, med_model, user_condition, image_path,
    vlm_metrics, med_metrics
):
    # Define all column names, including responses at end
    fieldnames = [
        "timestamp", "vlm_model", "med_model", "user_condition", "image_path",
        "vlm_total_duration", "vlm_load_duration", "vlm_prompt_eval_count", "vlm_prompt_eval_duration",
        "vlm_eval_count", "vlm_eval_duration", "vlm_tokens_per_second", "vlm_response_text",
        "med_total_duration", "med_load_duration", "med_prompt_eval_count", "med_prompt_eval_duration",
        "med_eval_count", "med_eval_duration", "med_tokens_per_second", "med_response_text"
    ]
    row = {
        "timestamp": datetime.now().isoformat(),
        "vlm_model": vlm_model,
        "med_model": med_model,
        "user_condition": user_condition,
        "image_path": image_path,
    }
    for i, m in enumerate(vlm_metrics):
        row[fieldnames[5+i]] = m
    for i, m in enumerate(med_metrics):
        row[fieldnames[13+i]] = m
    csv_file = os.path.join(os.path.dirname(__file__), "dietary_guidance_metrics.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def parse_vlm_output(vlm_result):
    edible_items = []
    try:
        for line in vlm_result.strip().split('\n'):
            if '.' in line and '-' in line:
                item = line.split('.', 1)[1].split('-', 1)[0].strip()
                if item: edible_items.append(item)
    except Exception as e:
        print("[ERROR] Failed to parse VLM output:", e)
        traceback.print_exc()
    return edible_items

def vlm_inference_ollama(image, prompt, image_path="", user_condition=""):
    metrics_out = [None] * 8
    try:
        print("Loading VLM...")
        img_base64 = [image_to_base64_jpeg(image)]
        data = {
            "model": VLM_MODEL,
            "prompt": prompt,
            "images": img_base64,
            "stream": False,
            "keep_alive": "60s"
        }
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=data).json()
        metrics_out = extract_ollama_metrics(response, is_chat=False)
        result = metrics_out[-1] or ""
        print("  [TIMER] VLM inference complete.")
        return result, metrics_out
    except Exception as e:
        print("[ERROR] VLM inference failed:", e)
        traceback.print_exc()
        return "", metrics_out

def medgemma_inference_ollama(messages):
    metrics_out = [None] * 8
    try:
        print("Loading MedGemma...")
        data = {
            "model": MED_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "120s"
        }
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data).json()
        metrics_out = extract_ollama_metrics(response, is_chat=True)
        result = metrics_out[-1] or ""
        print("  [TIMER] MedGemma inference complete.")
        return result, metrics_out
    except Exception as e:
        print("[ERROR] Medical LLM inference failed:", e)
        traceback.print_exc()
        return "", metrics_out

def detect_items(image, user_condition, image_path=""):
    if image is None or user_condition.strip() == "":
        return "", "", [], gr.update(visible=False), [None]*8, image_path
    vlm_raw, vlm_metrics = vlm_inference_ollama(image, VLM_PROMPT, image_path=image_path, user_condition=user_condition)
    edible_items = parse_vlm_output(vlm_raw)
    detected = ", ".join(edible_items)
    return vlm_raw, detected, edible_items, gr.update(visible=True), vlm_metrics, image_path

def correct_items(edible_items, user_add):
    user_add = user_add.strip()
    if user_add:
        additions = [item.strip() for item in user_add.split(",") if item.strip()]
        for item in additions:
            if item and item not in edible_items:
                edible_items.append(item)
    final_list = ", ".join(edible_items)
    return edible_items, final_list

def medgemma_guidance(user_condition, edible_items, image_path="", vlm_metrics=None):
    system_msg = {
        "role": "system",
        "content": (
            "You are MedGemma, an expert clinical nutrition and gastrointestinal dietary advisor. "
            "Your role is to provide clear, evidence-based dietary guidance for patients with diverticulitis, considering their current phase (flare or remission) and any other context given. "
            "Given a list of food, fruit, drink, or edible items—including any regional, less-known, or international foods—confidently classify each item as Safe, Unsafe, or Caution for this patient, based on the best medical knowledge for diverticulitis. "
            "For each item: "
            "• Use medical evidence about fiber, fat, seeds, skin, acidity, and food preparation relevant for diverticulitis. "
            "• If the food is unknown or highly regional, make your best assessment based on its ingredients or category (e.g., 'fermented rice cake' or 'local spiced pickle'). "
            "• Never skip or merge items: If unsure, label as 'Caution' and explain why. "
            "Output as a markdown table:\n"
            "| Food Item | Classification | Rationale |\n"
            "|---|---|---|\n"
            "After the table, provide a clear and practical summary of overall dietary advice for this meal, tailored to diverticulitis (note if the patient is in flare or remission, if specified).\n"
            "Example rows:\n"
            "| Food Item         | Classification | Rationale |\n"
            "|-------------------|---------------|-----------|\n"
            "| White rice        | Safe          | Low in fiber, gentle on the gut, well-tolerated during flare. |\n"
            "| Whole wheat bread | Caution       | May be high in fiber; better tolerated in remission phase. |\n"
            "| Spicy mango pickle| Unsafe        | Spicy, acidic, may trigger symptoms and cause irritation. |\n"
            "| Ragi dosa         | Caution       | Regional, usually high in fiber; introduce slowly and monitor tolerance. |\n"
            "| Bael fruit juice  | Safe          | Traditionally used for digestive health, low residue. |\n"
            "| Gondhoraj lemon   | Caution       | Regional citrus, may aggravate symptoms if patient is sensitive to acidity. |\n"
            "\n"
            "Classify the following items for this patient:"
        )
    }
    if not user_condition.strip():
        user_condition = "no reported digestive condition or symptoms"
    med_prompt = (
        f"A patient with {user_condition.strip()} has the following food, fruit, and drink items detected: "
        f"{', '.join(edible_items)}."
    )
    context_memory = [system_msg, {"role": "user", "content": med_prompt}]
    medgemma_result, med_metrics = medgemma_inference_ollama(context_memory)
    chat_history = [
        {"role": "user", "content": med_prompt},
        {"role": "assistant", "content": medgemma_result}
    ]
    if vlm_metrics is None:
        vlm_metrics = [None] * 8
    log_metrics_csv(
        VLM_MODEL, MED_MODEL, user_condition, image_path or "N/A",
        vlm_metrics, med_metrics
    )
    return medgemma_result, context_memory, chat_history

# ---- Gradio interface, now passing metrics and image path via states ----

with gr.Blocks() as demo:
    gr.Markdown("# 🥗 AI-powered Dietary Guidance for Diverticulitis Disease\nUpload a meal image and describe your current digestive condition (e.g., 'History of diverticulitis with stricture'). Review and correct detected foods, get safe/unsafe/caution advice, and chat with MedGemma!")

    with gr.Tab("Analyze Meal"):
        with gr.Row():
            image = gr.Image(type="pil", label="Upload Meal Image")
            user_condition = gr.Textbox(label="Describe your digestive condition", lines=2, placeholder="e.g., History of diverticulitis with occasional strictures, currently asymptomatic.")
        detect_btn = gr.Button("Detect Edible Items")
        vlm_raw = gr.Textbox(label="VLM Raw Output", visible=False)
        detected = gr.Textbox(label="Detected Edible Items", interactive=False)
        edible_items_state = gr.State([])
        review_row = gr.Row(visible=False)
        with review_row:
            user_add = gr.Textbox(label="Add any missed food/drink items (comma separated)")
            final_detected = gr.Textbox(label="Final Food/Drink List for Analysis", interactive=False)
            submit_btn = gr.Button("Get Dietary Advice")
    medgemma_output = gr.Textbox(label="MedGemma Dietary Guidance", lines=12, interactive=False)
    context_memory_state = gr.State([])
    chat_history_state = gr.State([])
    vlm_metrics_state = gr.State([None]*8)
    image_path_state = gr.State("N/A")

    def on_detect(image, user_condition):
        # If image is uploaded via file, it may have .name, else use 'N/A'
        image_path = getattr(image, 'name', 'N/A') if image is not None and hasattr(image, "name") else 'N/A'
        vlm_raw, detected, edible_items, review_row_out, vlm_metrics, img_path = detect_items(image, user_condition, image_path=image_path)
        return vlm_raw, detected, edible_items, review_row_out, "", "", "", [], [], vlm_metrics, img_path

    detect_btn.click(
        on_detect,
        inputs=[image, user_condition],
        outputs=[vlm_raw, detected, edible_items_state, review_row, user_add, final_detected, medgemma_output, context_memory_state, chat_history_state, vlm_metrics_state, image_path_state],
    )

    def on_correct(edible_items, user_add):
        return correct_items(edible_items, user_add)

    user_add.submit(
        on_correct,
        inputs=[edible_items_state, user_add],
        outputs=[edible_items_state, final_detected]
    )

    def on_submit(user_condition, edible_items, vlm_metrics, image_path):
        return medgemma_guidance(user_condition, edible_items, image_path=image_path, vlm_metrics=vlm_metrics)

    submit_btn.click(
        on_submit,
        inputs=[user_condition, edible_items_state, vlm_metrics_state, image_path_state],
        outputs=[medgemma_output, context_memory_state, chat_history_state]
    )

demo.queue().launch()

