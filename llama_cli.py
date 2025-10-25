#!/usr/bin/env python3
import json
import io
import os
import sys
import base64
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# --- CONFIGURATION ---
MODEL_GGUF_PATH = "./Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
CLIP_MODEL_PATH = "./mmproj-F16.gguf"

# --- LOAD MODEL ---
print("Loading Qwen2.5-VL model...", file=sys.stderr)
llm = None
chat_handler = None

if not os.path.exists(MODEL_GGUF_PATH) or not os.path.exists(CLIP_MODEL_PATH):
    print(f"❌ Model files not found:", file=sys.stderr)
    print(f"   {MODEL_GGUF_PATH}", file=sys.stderr)
    print(f"   {CLIP_MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    # First create the chat handler with the CLIP model
    chat_handler = Llava15ChatHandler(clip_model_path=CLIP_MODEL_PATH)
    
    # Then load the LLM with the chat handler
    llm = Llama(
        model_path=MODEL_GGUF_PATH,
        chat_handler=chat_handler,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False,
        logits_all=True  # Required for vision models
    )
    print("✅ Qwen2.5-VL Model loaded successfully\n", file=sys.stderr)
except Exception as e:
    print(f"❌ Error loading VLM: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


def image_to_data_uri(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI."""
    img_byte_arr = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


def process_query(prompt, image_path=None):
    """Process a query with optional image, just like Ollama."""
    
    if image_path:
        print(f"📷 Loading image: {image_path}", file=sys.stderr)
        try:
            pil_image = Image.open(image_path)
            image_uri = image_to_data_uri(pil_image)
            
            message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt}
                ]
            }
        except Exception as e:
            print(f"❌ Error loading image: {e}", file=sys.stderr)
            return
    else:
        message = {"role": "user", "content": prompt}
    
    print("🤖 Processing...\n", file=sys.stderr)
    
    try:
        response = llm.create_chat_completion(
            messages=[message],
            max_tokens=2048,
            temperature=0.0,
            stream=True
        )
        
        for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


def chat_mode():
    """Interactive chat mode - keeps conversation history."""
    conversation = []
    
    print("💬 Chat mode - Type 'exit' or 'quit' to leave")
    print("Usage: <prompt> - <image_path>\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            # Parse input for image path (Ollama style: prompt - path)
            image_path = None
            prompt = user_input
            
            if " - " in user_input:
                parts = user_input.split(" - ", 1)
                prompt = parts[0].strip()
                potential_path = parts[1].strip()
                
                if os.path.exists(potential_path):
                    image_path = potential_path
                else:
                    print(f"⚠️  File not found: {potential_path}", file=sys.stderr)
                    print("Continuing with text-only...\n", file=sys.stderr)
            
            # Build message
            if image_path:
                try:
                    pil_image = Image.open(image_path)
                    image_uri = image_to_data_uri(pil_image)
                    
                    message = {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                except Exception as e:
                    print(f"❌ Error loading image: {e}", file=sys.stderr)
                    continue
            else:
                message = {"role": "user", "content": prompt}
            
            conversation.append(message)
            
            response = llm.create_chat_completion(
                messages=conversation,
                max_tokens=2048,
                temperature=0.7,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            
            print("\n")
            conversation.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage (Ollama-style):")
        print('  python script.py "<prompt> - <image_path>"')
        print('  python script.py "<prompt>"')
        print("  python script.py chat")
        print("\nExamples:")
        print('  python script.py "extract all information from this business card and output it in json format(name, address, email, phone number, position, etc.) - business_card.jpg"')
        print('  python script.py "what is the capital of France?"')
        print("  python script.py chat")
        sys.exit(1)
    
    command = " ".join(sys.argv[1:])
    
    if command.lower() == "chat":
        chat_mode()
    else:
        # Parse Ollama-style input: "prompt - image_path"
        image_path = None
        prompt = command
        
        if " - " in command:
            parts = command.split(" - ", 1)
            prompt = parts[0].strip()
            potential_path = parts[1].strip()
            
            if os.path.exists(potential_path):
                image_path = potential_path
            else:
                print(f"⚠️  Warning: File not found: {potential_path}", file=sys.stderr)
                print("Continuing with text-only query...\n", file=sys.stderr)
        
        process_query(prompt, image_path)


if __name__ == "__main__":
    main()