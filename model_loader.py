import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import config

def load_model_and_tokenizer(model_path=config.MODEL_PATH):
    print(f">> [Loader] Đang chuẩn bị tải model: {model_path}")
    load_source = model_path

    # BẮT ĐẦU TẢI
    try:
        tokenizer = AutoTokenizer.from_pretrained(load_source)
        tokenizer.pad_token = tokenizer.eos_token
        
        # LOGIC TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG
        if torch.cuda.is_available():
            print(">> [System] Phát hiện GPU NVIDIA (Colab). Kích hoạt 4-bit...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                load_source,
                quantization_config=bnb_config,
                device_map="auto",
                # local_files_only=True if os.path.exists(model_path) else False 
                # ^ Dòng trên ép buộc dùng file local nếu đường dẫn tồn tại
            )
            device = "cuda"

        elif torch.backends.mps.is_available():
            print(">> [System] Phát hiện Mac M-Chip. Kích hoạt MPS...")
            model = AutoModelForCausalLM.from_pretrained(
                load_source,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            device = "mps"
            
        else:
            print(">> [System] Chạy bằng CPU...")
            model = AutoModelForCausalLM.from_pretrained(load_source)
            device = "cpu"

        print(">> [Loader] Tải model thành công!")
        return model, tokenizer, device

    except Exception as e:
        print(f"\n>> [FATAL ERROR] Lỗi khi tải model: {e}")
        print(">> Hãy kiểm tra lại đường dẫn Google Drive hoặc quyền truy cập.")
        raise e