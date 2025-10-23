# Local VLM Business Card Parser (Gradio Experiment)
This is a simple, local-only web app for testing the **Qwen2.5-VL-7B** model's ability to parse business cards.

It uses **Gradio** to create a simple "Upload Image" UI and shows the resulting JSON output. There is no API or database. This is purely for experimentation.
### How It Works
1. **VLM:** The ```Qwen2.5-VL-7B``` model is loaded locally using ```llama-cpp-python```.
2. **Gradio UI:** A simple web interface is created using the ```gradio``` library.
3. **Parsing:** You upload an image. The image is sent to the VLM with a prompt asking for JSON.
4. **Output:** The raw JSON output from the model is displayed in a JSON component.
### Setup Instructions
1. **System Dependencies**
**No Tesseract is needed!** You only need Python.
2. **Python Environment**
    1. It's highly recommended to use a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate
    ```
    On Windows:
    ```
    venv\Scripts\activate
    ```
    2. Install the required Python libraries (note the addition of ```gradio```):
    ```
    pip install gradio pillow llama-cpp-python
    ```
3. **Download the Qwen VLM (CRITICAL: 2 FILES)**
To run this VLM locally, you must download **two** files. Based on your screenshot, these are the correct files:
    1. Go to Hugging Face: Find the repository with the files (e.g., ```unsloth/Qwen2.5-VL-7B-Instruct-GGUF``` or similar).
    2. **Download BOTH files:**
        * **The Model:** ```Qwen2.5-VL-7B-instruct-UD-Q4_K_XL.gguf``` (4.79 GB)
        * **The Projector:** ```mmproj-F16.gguf``` (1.35 GB)
    3. Place both downloaded ```.gguf``` files in the same directory as your ```qwen_gradio_parser.py``` file.
    4. The script is already configured to look for these exact filenames.
### Running the App
1. With your virtual environment active, run the Python script from your terminal:
```
python qwen_gradio_parser.py
```
2. The script will load the VLM into memory (this will take several seconds and use ~6-7 GB of RAM).
3. Once loaded, it will print a URL in your terminal, typically:Running on local URL:  ```http://127.0.0.1:7860```
4. Open that URL in your browser.
5. You will see the Gradio interface. Upload a business card image and see the JSON output directly!