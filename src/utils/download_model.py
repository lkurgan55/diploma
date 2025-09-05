from transformers import AutoTokenizer, AutoModelForCausalLM
import os

HF_TOKEN = os.getenv("HF_TOKEN") or ''
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LOCAL_DIR = "./models/qwen2.5-3B-Instruct"

print(f"Download {MODEL_ID} in {LOCAL_DIR}...")

# Завантаження і збереження токенізатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(LOCAL_DIR)

# Завантаження і збереження самої моделі
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.save_pretrained(LOCAL_DIR)

print("✅ Succesfully downloaded and saved the model and tokenizer.")
