import gradio as gr
import json
import io
import base64
from PIL import Image
import ollama

# --- 1. CONFIGURATION ---
MODEL_NAME = "qwen2.5vl:3b-q4_K_M"  # Your local Ollama model

# --- 2. Helper Function ---
def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- 3. Core Parsing Function ---
def parse_business_card(pil_image):
    if pil_image is None:
        return {"error": "Please upload an image."}

    # Convert image to base64
    try:
        img_b64 = image_to_base64(pil_image)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    # Prompt
    '''
    user_prompt = (
        f"![business_card](data:image/jpeg;base64,{img_b64})\n\n"
        "You are an expert assistant that analyzes business card images. "
        "Your task is to extract the contact information and return *ONLY* "
        "a single, valid JSON object. Do not add any text, explanations, or markdown formatting before or after the JSON.\n\n"
        "Use this JSON schema (set missing fields to null):\n"
        "{\n"
        '  \"name\": \"...\",\n'
        '  \"title\": \"...\",\n'
        '  \"company\": \"...\",\n'
        '  \"phone\": \"...\",\n'
        '  \"email\": \"...\",\n'
        '  \"website\": \"...\",\n'
        '  \"address\": \"...\",\n'
        '  \"miscellaneous\": \"...\"\n'
        "}\n\n"
        "Respond with *only* the populated JSON object."
    )
    '''
    user_prompt = f"""
    You are a precise information extraction model. Carefully look at the attached business card image below.

    ![business_card](data:image/jpeg;base64,{img_b64})

    Extract the contact information **ONLY** from this image. Do not use any fabricated or default data.

    Return a single, well-formed JSON object using exactly this schema:
    {{
    "name": "...",
    "title": "...",
    "company": "...",
    "phone": "...",
    "email": "...",
    "website": "...",
    "address": "...",
    "miscellaneous": "..."
    }}

    Rules:
    - Fill fields with actual text from the card.
    - If a field is missing, set it to null.
    - Do not add extra commentary, explanations, or markdown.
    - Return ONLY valid JSON.
    """

    # Inference


    try:
        print(f"Image base64 length: {len(img_b64)}")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_prompt}],
            options={
                "temperature": 0.0,
                "num_predict": 1024,
                "top_p": 0.1,
                "repeat_penalty": 1.05
            }
        )

        json_text = response["message"]["content"].strip()

        # Remove possible markdown wrappers
        if json_text.startswith("```json"):
            json_text = json_text[7:-3].strip()

    except Exception as e:
        return {"error": f"Ollama inference failed: {e}"}

    # Parse JSON
    try:
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        print(f"Failed to parse model output:\n{json_text}")
        return {"error": "Failed to parse model output as JSON.", "raw_output": json_text}

# --- 4. Gradio UI ---
DESCRIPTION = """
## Qwen2.5VL 3B Business Card Parser  
Upload a business card image to extract structured contact info using your local Ollama model.
"""

if __name__ == "__main__":
    gr.Interface(
        fn=parse_business_card,
        inputs=gr.Image(type="pil", label="Upload Business Card"),
        outputs=gr.JSON(label="Extracted Information"),
        title="Ollama Business Card Parser (Qwen2.5VL 3B)",
        description=DESCRIPTION,
        allow_flagging="never",
    ).launch(share=False)
