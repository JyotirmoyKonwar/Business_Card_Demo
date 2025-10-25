import gradio as gr
import json
import os
from PIL import Image
from llama_cpp import Llama
import pytesseract

# Optional: Specify tesseract path if not in system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux

# --- 1. CONFIGURATION ---

MODEL_GGUF_PATH = "./Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
CHAT_FORMAT = "chatml"  # For text-only LLM usage

# --- 2. LOAD MODEL (Text-only mode) ---

llm = None

if not os.path.exists(MODEL_GGUF_PATH):
    print("❌ Model GGUF file not found.")
    print(f"Check: {MODEL_GGUF_PATH}")
else:
    try:
        # Load as text-only model (no vision projector needed)
        llm = Llama(
            model_path=MODEL_GGUF_PATH,
            chat_format=CHAT_FORMAT,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False
        )
        print("✅ Qwen2.5 model loaded successfully (text-only mode).")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        llm = None

# Test Tesseract
try:
    pytesseract.get_tesseract_version()
    print("✅ Tesseract OCR is available.")
except Exception as e:
    print(f"❌ Tesseract OCR not found: {e}")
    print("Install: apt-get install tesseract-ocr (Linux) or brew install tesseract (Mac)")
    print("Or download from: https://github.com/UB-Mannheim/tesseract/wiki")


# --- 3. OCR Extraction ---

def extract_text_from_image(pil_image):
    """Extract text from image using Tesseract OCR."""
    try:
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(pil_image, config='--psm 6')
        return text.strip()
    except Exception as e:
        return f"OCR Error: {e}"


# --- 4. Business Card Parser ---

def parse_business_card(pil_image):
    if pil_image is None:
        return {"error": "Please upload an image."}
    
    # Step 1: Extract text using OCR
    print("\n" + "="*60)
    print("STEP 1: Extracting text with Tesseract OCR...")
    print("="*60)
    
    ocr_text = extract_text_from_image(pil_image)
    
    if not ocr_text or "OCR Error" in ocr_text:
        return {"error": "Failed to extract text from image.", "details": ocr_text}
    
    print(f"📄 Extracted text:\n{ocr_text}\n")
    print("="*60 + "\n")
    
    # Step 2: Use LLM to structure the text into JSON
    if llm is None:
        # Fallback: return raw OCR text if LLM not available
        return {
            "error": "LLM not loaded - returning raw OCR text",
            "raw_ocr_text": ocr_text
        }
    
    print("STEP 2: Structuring text with LLM...")
    print("="*60)
    
    structuring_prompt = f"""You are an assistant that organizes business card information.

Below is text extracted from a business card via OCR:

---
{ocr_text}
---

Analyze this text and extract the following information into a JSON object:

{{
  "name": "person's full name",
  "title": "job title or position",
  "company": "company or organization name",
  "phone": "phone number(s)",
  "email": "email address",
  "website": "website URL",
  "address": "physical address",
  "miscellaneous": "any other relevant information"
}}

Rules:
1. Extract information only from the provided OCR text
2. Use empty string "" if a field is not found
3. Clean up any OCR errors if obvious (e.g., "emai1" → "email")
4. Combine multi-line addresses into a single string
5. Return ONLY the JSON object, no explanations

JSON:"""

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": structuring_prompt}
            ],
            max_tokens=2048,
            temperature=0.0,
            stop=["</s>", "<|im_end|>", "<|endoftext|>", "\n\n\n"]
        )
        
        text = response["choices"][0]["message"]["content"].strip()
        print(f"📝 LLM response:\n{text}\n")
        print("="*60 + "\n")
        
        # Clean up response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Extract JSON
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]
        
        # Parse JSON
        parsed = json.loads(text)
        
        # Add OCR text for reference
        parsed["_raw_ocr_text"] = ocr_text
        
        return parsed
        
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse JSON from LLM",
            "raw_ocr_text": ocr_text,
            "llm_output": text,
            "parse_error": str(e)
        }
    except Exception as e:
        return {
            "error": f"LLM structuring failed: {e}",
            "raw_ocr_text": ocr_text
        }


# --- 5. Gradio App ---

DESCRIPTION = """
### 🔍 Hybrid Business Card Parser (Tesseract OCR + LLM)

**How it works:**
1. **Tesseract OCR** extracts all text from the business card image
2. **Qwen2.5 LLM** structures the extracted text into organized JSON

**Requirements:**
- Tesseract OCR must be installed on your system
- Qwen2.5 model GGUF file in the same directory

**Advantages of this approach:**
- More reliable text extraction than pure VLM
- Works with any LLM (no vision capability needed)
- Better OCR accuracy for business cards
"""

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 HYBRID BUSINESS CARD PARSER")
    print("="*60)
    
    # Check dependencies
    tesseract_ok = False
    try:
        pytesseract.get_tesseract_version()
        tesseract_ok = True
        print("✅ Tesseract OCR: Available")
    except:
        print("❌ Tesseract OCR: NOT FOUND")
        print("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    if llm:
        print("✅ LLM: Loaded")
    else:
        print("⚠️  LLM: Not loaded (will return raw OCR only)")
    
    print("="*60 + "\n")
    
    if not tesseract_ok:
        print("\n⚠️  WARNING: Tesseract not found. Please install it first.")
        print("Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("Mac: brew install tesseract")
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n")
    
    demo = gr.Interface(
        fn=parse_business_card,
        inputs=gr.Image(type="pil", label="Upload Business Card"),
        outputs=gr.JSON(label="Extracted Information"),
        title="🔍 Hybrid Business Card Parser",
        description=DESCRIPTION,
        examples=[],
        allow_flagging="never",
    )
    
    print("🚀 Launching Gradio interface...")
    demo.launch(share=False)