import gradio as gr
import json
import io
import os
import base64
from PIL import Image
from llama_cpp import Llama

# --- 1. CONFIGURATION ---

# IMPORTANT: Update these paths to match the files you downloaded
# From your screenshot:
MODEL_GGUF_PATH = "./Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
CLIP_MODEL_PATH = "./mmproj-F16.gguf"

# CRITICAL: Change chat format to "qwen2"
CHAT_FORMAT = "qwen2"

# --- 2. Load Local VLM ---

llm = None
chat_handler = None

if not os.path.exists(MODEL_GGUF_PATH) or not os.path.exists(CLIP_MODEL_PATH):
    print("Error: Model GGUF or CLIP projector file not found.")
    print(f"Make sure '{MODEL_GGUF_PATH}' and '{CLIP_MODEL_PATH}' exist.")
else:
    try:
        llm = Llama(
            model_path=MODEL_GGUF_PATH,
            chat_format=CHAT_FORMAT,
            n_ctx=4096,      # Context window
            n_threads=8,     # Adjust to your CPU cores
            n_gpu_layers=0,  # Run 100% on CPU
            verbose=False    # Suppress llama.cpp logs
        )
        # This is the key step to load the VLM capabilities
        chat_handler = llm.chat_handler(
            clip_model_path=CLIP_MODEL_PATH
        )
        print("Qwen2.5-VL Model loaded successfully.")
    except Exception as e:
        print(f"Error loading VLM: {e}")
        llm = None
        chat_handler = None

# --- 3. Helper Function ---

def image_to_data_uri(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI."""
    img_byte_arr = io.BytesIO()
    # Ensure image is RGB for consistency
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

# --- 4. Core Parsing Function for Gradio ---

def parse_business_card(pil_image):
    """
    Takes a PIL image, sends it to the VLM, and returns the extracted JSON.
    """
    if llm is None or chat_handler is None:
        return {"error": "VLM model is not loaded. Check server logs."}
    
    if pil_image is None:
        return {"error": "Please upload an image."}

    # 1. Convert image to data URI
    try:
        image_uri = image_to_data_uri(pil_image)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}
    
    # 2. Create the VLM prompt
    user_prompt = (
        "You are an expert assistant that analyzes business card images. "
        "Your task is to extract the contact information and return *ONLY* "
        "a single, valid JSON object. Do not add any text, "
        "explanations, or markdown formatting before or after the JSON.\n"
        "Analyze the attached business card image and extract the contact details. "
        "Use the following JSON schema and set fields to null if not found:\n"
        "{\n"
        '  "name": "...",\n'
        '  "title": "...",\n'
        '  "company": "...",\n'
        '  "phone": "...",\n'
        '  "email": "...",\n'
        '  "website": "...",\n'
        '  "address": "..."\n'
        '  "miscellaneous": "..."\n'
        "}\n\n"
        "Respond with *only* the populated JSON object."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    # 3. Perform VLM Inference
    try:
        response = chat_handler.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.0 # Set to 0.0 for deterministic JSON output
        )
        json_text = response['choices'][0]['message']['content'].strip()
        
        # Clean the output in case the model adds markdown
        if json_text.startswith("```json"):
            json_text = json_text[7:-3].strip()
        
    except Exception as e:
        return {"error": f"VLM inference failed: {e}"}
    
    # 4. Parse JSON and return
    try:
        parsed_data = json.loads(json_text)
        return parsed_data
    except Exception as e:
        print(f"Failed to parse VLM JSON output. Raw output: {json_text}")
        return {"error": "Failed to parse VLM output as JSON.", "raw_output": json_text}

# --- 5. Create and Launch Gradio App ---

DESCRIPTION = """
## Local Qwen2.5-VL Business Card Parser
Upload a business card image to see the VLM extract the information in real-time.
This demo runs 100% locally on your CPU.
"""

if __name__ == "__main__":
    if llm is None or chat_handler is None:
        print("\n--- FATAL ERROR ---")
        print("VLM not loaded. The Gradio app cannot start.")
        print("Please download the GGUF model and CLIP projector file.")
        print(f"1. {MODEL_GGUF_PATH}")
        print(f"2. {CLIP_MODEL_PATH}")
        print("-------------------\n")
    else:
        print("Launching Gradio Interface...")
        demo = gr.Interface(
            fn=parse_business_card,
            inputs=gr.Image(type="pil", label="Upload Business Card"),
            outputs=gr.JSON(label="Extracted Information"),
            title="Local VLM Business Card Parser",
            description=DESCRIPTION,
            allow_flagging="never"
        )
        
        # Share=True creates a public link for 72h.
        # Set to False if you only want to run it on localhost.
        demo.launch(share=False)
