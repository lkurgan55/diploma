from transformers import AutoTokenizer, AutoModelForCausalLM
import os

HF_TOKEN = os.getenv("HF_TOKEN") or ''
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_DIR = "./models/llama-3-8b-instruct"

print(f"Завантаження моделі {MODEL_ID} у {LOCAL_DIR}...")

# Завантаження і збереження токенізатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
tokenizer.save_pretrained(LOCAL_DIR)

# Завантаження і збереження самої моделі
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model.save_pretrained(LOCAL_DIR)

print("✅ Модель успішно збережено.")
