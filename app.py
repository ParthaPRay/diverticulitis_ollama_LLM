import gradio as gr
from ollama import Client
from PIL import Image
from io import BytesIO
import traceback
import time

OLLAMA_HOST = "http://localhost:11434"
VLM_MODEL = "gemma3:12b"
MED_MODEL = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"

OLLAMA_CLIENT = Client(host=OLLAMA_HOST)

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

def vlm_inference_ollama(image, prompt):
    try:
        # Load VLM model
        print("Loading VLM...")
        OLLAMA_CLIENT.generate(model=VLM_MODEL, prompt="", stream=False, keep_alive="60s")
        t0 = time.time()
        # In-memory JPEG encoding
        buf = BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        print("  [TIMER] Image encode:", time.time()-t0, "sec")
        t1 = time.time()
        response = OLLAMA_CLIENT.generate(
            model=VLM_MODEL,
            prompt=prompt,
            images=[img_bytes],
            stream=False,
            keep_alive="60s"
        )
        print("  [TIMER] VLM inference:", time.time()-t1, "sec")
        print("  [TIMER] VLM TOTAL:", time.time()-t0, "sec")
        result = response.get('response', '').strip()
        # Unload VLM model
        OLLAMA_CLIENT.generate(model=VLM_MODEL, prompt="", stream=False, keep_alive=0)
        print("Unloaded VLM.")
        return result
    except Exception as e:
        print("[ERROR] VLM inference failed:", e)
        traceback.print_exc()
        return ""

def medgemma_inference_ollama(messages):
    try:
        # Load MedGemma model
        print("Loading MedGemma...")
        OLLAMA_CLIENT.generate(model=MED_MODEL, prompt="", stream=False)
        t0 = time.time()
        # Build chat prompt for MedGemma
        chat_prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                chat_prompt += f"(System) {msg['content']}\n"
            elif msg['role'] == 'user':
                chat_prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                chat_prompt += f"MedGemma: {msg['content']}\n"
        response = OLLAMA_CLIENT.generate(
            model=MED_MODEL,
            prompt=chat_prompt,
            stream=False,
            keep_alive="120s"
        )
        print("  [TIMER] MedGemma inference:", time.time()-t0, "sec")
        result = response.get('response', '').strip()
        # Unload MedGemma model
        OLLAMA_CLIENT.generate(model=MED_MODEL, prompt="", stream=False, keep_alive=0)
        print("Unloaded MedGemma.")
        return result
    except Exception as e:
        print("[ERROR] Medical LLM inference failed:", e)
        traceback.print_exc()
        return ""

def detect_items(image, user_condition):
    if image is None or user_condition.strip() == "":
        return "", "", [], gr.update(visible=False)
    vlm_raw = vlm_inference_ollama(image, VLM_PROMPT)
    edible_items = parse_vlm_output(vlm_raw)
    detected = ", ".join(edible_items)
    return vlm_raw, detected, edible_items, gr.update(visible=True)

def correct_items(edible_items, user_add):
    user_add = user_add.strip()
    if user_add:
        additions = [item.strip() for item in user_add.split(",") if item.strip()]
        for item in additions:
            if item and item not in edible_items:
                edible_items.append(item)
    final_list = ", ".join(edible_items)
    return edible_items, final_list

def medgemma_guidance(user_condition, edible_items):
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
    medgemma_result = medgemma_inference_ollama(context_memory)
    chat_history = [
        {"role": "user", "content": med_prompt},
        {"role": "assistant", "content": medgemma_result}
    ]
    return medgemma_result, context_memory, chat_history

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

    def on_detect(image, user_condition):
        return detect_items(image, user_condition) + ("", "", [], [], [])

    detect_btn.click(
        on_detect,
        inputs=[image, user_condition],
        outputs=[vlm_raw, detected, edible_items_state, review_row, user_add, final_detected, medgemma_output, context_memory_state, chat_history_state],
    )

    def on_correct(edible_items, user_add):
        return correct_items(edible_items, user_add)

    user_add.submit(
        on_correct,
        inputs=[edible_items_state, user_add],
        outputs=[edible_items_state, final_detected]
    )

    def on_submit(user_condition, edible_items):
        return medgemma_guidance(user_condition, edible_items)

    submit_btn.click(
        on_submit,
        inputs=[user_condition, edible_items_state],
        outputs=[medgemma_output, context_memory_state, chat_history_state]
    )

demo.queue().launch()

