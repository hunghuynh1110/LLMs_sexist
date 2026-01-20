import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def load_model_and_tokenizer(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    print(f">> [Loader] Bắt đầu tải model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Llama 3 cần pad_token_id là eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- LOGIC TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG ---
    
    # TRƯỜNG HỢP 1: CHẠY TRÊN COLAB (NVIDIA GPU)
    if torch.cuda.is_available():
        print(">> [System] Phát hiện GPU NVIDIA (Colab/Cluster). Kích hoạt chế độ 4-bit Quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        device = "cuda"

    # TRƯỜNG HỢP 2: CHẠY TRÊN MAC (APPLE SILICON)
    elif torch.backends.mps.is_available():
        print(">> [System] Phát hiện Apple Silicon (M-Chip). Kích hoạt chế độ MPS...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        device = "mps" # Metal Performance Shaders
        
    # TRƯỜNG HỢP 3: CPU (CHẬM)
    else:
        print(">> [System] Không thấy GPU. Chạy bằng CPU (Sẽ rất chậm)...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        device = "cpu"

    print(">> [Loader] Tải model thành công!")
    return model, tokenizer, device