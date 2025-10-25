import gradio as gr
import json
import io
import ollama
from PIL import Image

# --- 1. CONFIGURATION ---

# Define the Ollama model you want to use
# This is CORRECT. 'ollama run' uses this name, so will 'ollama.chat'.
MODEL_NAME = "moondream" 

# --- 2. Helper Function ---

def pil_to_bytes(image: Image.Image) -> bytes:
    """
    Convert a PIL Image to bytes in JPEG format for Ollama.
    """
    img_byte_arr = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# --- 3. Core Parsing Function for Gradio ---

def parse_business_card(pil_image):
    """
    Takes a PIL image, sends it to Ollama/moondream, and returns the extracted JSON.
    """
    if pil_image is None:
        return {"error": "Please upload an image."}

    try:
        image_bytes = pil_to_bytes(pil_image)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}
    
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
        '  "address": "...",\n'
        '  "miscellaneous": "..."\n'
        "}\n\n"
        "Respond with *only* the populated JSON object."
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME, # This will correctly use 'moondream'
            messages=[
                {
                    'role': 'user',
                    'content': user_prompt,
                    'images': [image_bytes] 
                }
            ],
            options={
                'temperature': 0.0
            }
        )
        json_text = response['message']['content'].strip()
        
        # Clean the output in case the model adds markdown
        if json_text.startswith("```json"):
            json_text = json_text[7:-3].strip()
        elif json_text.startswith("`"):
             json_text = json_text.strip('` \n')

    except Exception as e:
        return {"error": f"Ollama inference failed: {e}. Is the Ollama server running?"}
    
    try:
        parsed_data = json.loads(json_text)
        return parsed_data
    except Exception as e:
        print(f"Failed to parse VLM JSON output. Raw output: {json_text}")
        return {"error": "Failed to parse VLM output as JSON.", "raw_output": json_text}

# --- 4. Health Check Function (FIXED LOGIC) ---

def check_model_availability(model_name):
    """Check if Ollama is running and the specified model is pulled."""
    try:
        response = ollama.list()
        
        if 'models' not in response or not isinstance(response.get('models'), list):
            print(f"--- Error ---")
            print("Ollama returned an unexpected response. Cannot verify models.")
            return False
            
        models_info = response['models']
        
        # --- THIS IS THE FIX ---
        # We check if any model name *starts with* the base name.
        # This correctly handles "moondream:latest" matching "moondream".
        model_found = any(
            isinstance(m, dict) and 'name' in m and m['name'].startswith(model_name)
            for m in models_info
        )
        # ---------------------
        
        if not model_found:
            print(f"--- Error ---")
            # The error message is now more accurate
            print(f"No model found that starts with the name '{model_name}'.")
            print(f"Please pull the model by running: ollama pull {model_name}")
            print("-------------")
            return False
            
        print(f"Ollama server is running and model '{model_name}' (or a variant) is available.")
        return True
        
    except Exception as e:
        print(f"--- Error Connecting to Ollama ---")
        print(f"Error: {e}")
        print("Please ensure the Ollama server is running.")
        print("(Start Ollama.app or run 'ollama serve' in your terminal).")
        print("---------------------------------")
        return False

# --- 5. Create and Launch Gradio App ---

DESCRIPTION = """
## Ollama (moondream) Business Card Parser
Upload a business card image to see the VLM extract the information in real-time.
This demo runs using a local Ollama server.
"""

if __name__ == "__main__":
    if check_model_availability(MODEL_NAME):
        print("Launching Gradio Interface...")
        demo = gr.Interface(
            fn=parse_business_card,
            inputs=gr.Image(type="pil", label="Upload Business Card"),
            outputs=gr.JSON(label="Extracted Information"),
            title="Ollama VLM Business Card Parser",
            description=DESCRIPTION,
            allow_flagging="never"
        )
        
        demo.launch(share=False)
    else:
        print("\n--- FATAL ERROR ---")
        print("Gradio app cannot start. Please resolve the Ollama issues listed above.")
        print("-------------------\n")