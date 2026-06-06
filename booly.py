import ollama,os,easyocr,numpy as np,re,subprocess,json,torch,pyttsx3
from PIL import ImageGrab
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import speech_recognition as sr,threading
from faster_whisper import WhisperModel

if not os.path.exists('./ollama_model'):
    SentenceTransformer('all-MiniLM-L6-v2').save('./ollama_model')

os.environ["HF_HUB_OFFLINE"] = "1"

if os.path.exists('codes_db'):
    code_db = Chroma(persist_directory='codes_db',embedding_function=HuggingFaceEmbeddings(model_name='./ollama_model'))
else:
    code_db=Chroma(embedding_function=HuggingFaceEmbeddings(model_name='./ollama_model'),persist_directory='codes_db')
    with open('codes.json', 'r', encoding='utf-8') as f:
        codes_data = json.load(f)

    for item in codes_data:
        code_db.add_documents([Document(page_content=item['task'], metadata={"code": item['code']})])

def booly(sual,ocr_promt=None):
    snippet_results = code_db.similarity_search(sual, k=2)
    expert_code_context = ""
    for i, doc in enumerate(snippet_results):
        code = doc.metadata.get('code', 'No code found')
        expert_code_context += f"\n--- Expert Template {i+1} ---\n{code}\n"

    print(f"Expert Code Snippets:\n{expert_code_context}\n")

    final_sual=sual
    cavab= ollama.chat(model='gemma2:2b',
                        messages=[{'role': 'system',
                            'content': (f"""You are Booly, a computer automation agent. Your mission is to generate Python code based ONLY on the EXPERT TEMPLATES provided.

### 🛑 CRITICAL RULES:
1. COORDINATES: If the task requires clicking or typing, but no OCR DATA, respond with the word GET_SCREEN_MAP
2. PRIORITY: If there is a “web browser” template for the task, use it. Do not use pyautogui for web navigation.
3. NO HALLUCINATION: Use only libraries and functions that are in the EXPERT TEMPLATES.

### 📦 EXPERT TEMPLATES:
{expert_code_context if expert_code_context else "No expert templates found."}

### 📥 RESPONSE FORMAT:
Please provide a very brief explanation, then provide the code strictly inside a Markdown Python block: ```python\n# Your code here\n```

"""

                
            )},{"role": "user",
                                    "content": final_sual}],
                        options={'temperature': 0.3,       
                                'top_p': 0.85,            
                                'repeat_penalty': 1.15,   
                                'num_ctx': 2048,          
                                'num_predict': 1024,
                                'num_thread': 8})

    return cavab['message']['content'].strip()



engine = pyttsx3.init()
engine.setProperty('rate', 170) 
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

stt_model = WhisperModel("base", device="cpu", compute_type="int8")
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.8  
recognizer.dynamic_energy_threshold = True
while True:
    with sr.Microphone() as source:
        print("🎙️ Dinlənilir...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
        except sr.WaitTimeoutError:
            print("⚠️ Heç bir səs eşidilmədi.")
            continue
    segments, _ = stt_model.transcribe("temp_audio.wav", beam_size=5, language="en")
    
    text = "".join([segment.text for segment in segments]).strip()
    
    # Müvəqqəti faylı silirik
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    sual = input("You: ")
    
    if sual.lower() in ['exit', 'quit', 'q']:
        break
    if sual.lower() in '/clr code_db':
        if os.path.exists('codes_db'):
            os.remove('codes_db')
    if sual.lower() in '/clr vector_db':
        if os.path.exists('chroma_db'):
            os.remove('chroma_db')
    ai_cavab = booly(sual)
    if 'GET_SCREEN_MAP' in ai_cavab:
        ai_cavab = booly(sual,ocr_promt=ocr_image())


    if '```python' in ai_cavab:
        code_match = re.search(r"```python\n(.*?)\n```", ai_cavab.lower(), re.DOTALL | re.IGNORECASE)
    
        if code_match:
            raw_code = code_match.group(1).strip()
            clean_code = re.sub(r"^python\n", "", raw_code)
            library_map = {
        'os': 'import os',
        'subprocess': 'import subprocess',
        'pyautogui': 'import pyautogui',
        'requests': 'import requests',
        'json': 'import json',
        'shutil': 'import shutil',
        'time': 'import time',
        'psutil': 'import psutil',
        'webbrowser': 'import webbrowser',
        'cv2': 'import cv2',
        'numpy': 'import numpy as np',
        'pd.': 'import pandas as pd',
        'plt.': 'import matplotlib.pyplot as plt',
        'Document': 'from langchain_core.documents import Document'
        }

            found_imports = []
            for key, import_statement in library_map.items():
                if key in clean_code and import_statement not in clean_code:
                    found_imports.append(import_statement)

            if found_imports:
                clean_imports = "\n".join(found_imports)
                full_code = f"{clean_imports}\n\n{clean_code}"

            temp_file = "temp_executor.py"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(full_code if found_imports else clean_code)
        
            try:
                result = subprocess.run(
                    ["python", temp_file], 
                    capture_output=True, 
                    text=True, 
                    timeout=15 
                )
                if result.returncode == 0:
                    if result.stdout.strip():
                            ai_cavab += f"\n\n✅ Kod icra edildi. Çıktı:\n{result.stdout.strip()}"
                    else:
                        
                        print("✅ Uğurla icra edildi, ancak heç bir çıktı vermedi.")
                        
                else:
                    print(f"❌ Kod xətası:\n{result.stderr}")
                
            except subprocess.TimeoutExpired:
                print("⚠️ Xəta: Kod çox uzun çəkdi (Timeout).")
            except Exception as e:
                print(f"⚠️ Gözlənilməz xəta: {e}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            print("ℹ️ Model cavabında icra edilə bilən kod bloku tapılmadı.")
    print(f"\nBooly: {ai_cavab}\n",flush=True)
