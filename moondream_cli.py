#!/usr/bin/env python3
import json
import io
import os
import sys
import base64
from PIL import Image
from llama_cpp import Llama
# CHANGED: Imported MoondreamChatHandler
from llama_cpp.llama_chat_format import MoondreamChatHandler

# --- CONFIGURATION ---
# CHANGED: Updated file paths for moondream2
MODEL_GGUF_PATH = "./moondream2-text-model-f16.gguf"
CLIP_MODEL_PATH = "./moondream2-mmproj-f16.gguf"

# --- LOAD MODEL ---
# CHANGED: Updated print statement
print("Loading Moondream2 model...", file=sys.stderr)
llm = None
chat_handler = None

if not os.path.exists(MODEL_GGUF_PATH) or not os.path.exists(CLIP_MODEL_PATH):
    print(f"❌ Model files not found:", file=sys.stderr)
    print(f"   {MODEL_GGUF_PATH}", file=sys.stderr)
    print(f"   {CLIP_MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    # CHANGED: Using MoondreamChatHandler
    chat_handler = MoondreamChatHandler(clip_model_path=CLIP_MODEL_PATH, verbose=False)
    
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
    # CHANGED: Updated print statement
    print("✅ Moondream2 Model loaded successfully\n", file=sys.stderr)
except Exception as e:
    print(f"❌ Error loading VLM: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


def image_to_data_uri(image_path: str) -> str:
    """Convert an image file to a base64 data URI."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 1024px on longest side)
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr = img_byte_arr.getvalue()
            base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
            
            return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        raise Exception(f"Failed to process image: {e}")


def process_query(prompt, image_path=None):
    """Process a query with optional image, just like Ollama."""
    
    messages = []
    
    if image_path:
        print(f"📷 Loading image: {image_path}", file=sys.stderr)
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}", file=sys.stderr)
            return
        
        try:
            image_uri = image_to_data_uri(image_path)
            print(f"✅ Image loaded successfully", file=sys.stderr)
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt}
                ]
            })
        except Exception as e:
            print(f"❌ Error loading image: {e}", file=sys.stderr)
            return
    else:
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    print("🤖 Processing with VLM...\n", file=sys.stderr)
    
    try:
        # Get initial token count
        initial_tokens = llm.n_tokens
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
            stream=True,
            # CHANGED: Updated stop token for Moondream
            stop=["</s>"]
        )
        
        generated_tokens = 0
        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    generated_tokens += 1
        
        print("\n")
        
        # Calculate token usage
        total_tokens = llm.n_tokens
        # Note: Token calculation might be slightly different, but logic is the same
        prompt_tokens = total_tokens - initial_tokens - generated_tokens
        
        print(f"\n📊 Token Usage:", file=sys.stderr)
        print(f"   Prompt tokens: {prompt_tokens}", file=sys.stderr)
        print(f"   Generated tokens: {generated_tokens}", file=sys.stderr)
        print(f"   Total tokens: {total_tokens}", file=sys.stderr)
        print(f"   Context used: {total_tokens}/{llm.n_ctx()} ({100*total_tokens/llm.n_ctx():.1f}%)\n", file=sys.stderr)
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}", file=sys.stderr)
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
            
            # Look for " - " separator
            if " - " in user_input:
                parts = user_input.rsplit(" - ", 1)  # Split from right to handle paths with dashes
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
                    image_uri = image_to_data_uri(image_path)
                    
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
                stream=True,
                # CHANGED: Updated stop token for Moondream
                stop=["</s>"]
            )
            
            full_response = ""
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
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


def main():
    if len(sys.argv) < 2:
        print("Usage (Ollama-style):")
        print('  python script.py "<prompt> - <image_path>"')
        print('  python script.py "<prompt>"')
        print("  python script.py chat")
        print("\nExamples:")
        print('  python script.py "extract info - card.jpg"')
        print('  python script.py "what is AI?"')
        print("  python script.py chat")
        print("\nIMPORTANT: Note the space before and after the dash: ' - '")
        sys.exit(1)
    
    command = " ".join(sys.argv[1:])
    
    if command.lower() == "chat":
        chat_mode()
    else:
        # Parse Ollama-style input: "prompt - image_path"
        image_path = None
        prompt = command
        
        # Look for " - " separator (with spaces)
        if " - " in command:
            parts = command.rsplit(" - ", 1)  # Split from right to handle paths with dashes
            prompt = parts[0].strip()
            potential_path = parts[1].strip()
            
            if os.path.exists(potential_path):
                image_path = potential_path
            else:
                print(f"❌ Error: File not found: {potential_path}", file=sys.stderr)
                print(f"Make sure the path is correct and use ' - ' (space-dash-space) as separator\n", file=sys.stderr)
                sys.exit(1)
        
        process_query(prompt, image_path)


if __name__ == "__main__":
    main()
