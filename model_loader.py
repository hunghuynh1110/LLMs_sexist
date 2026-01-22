import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import config

def load_model_and_tokenizer(model_path=config.MODEL_PATH):
    # KIỂM TRA NGUỒN MODEL
    if os.path.exists(model_path):
        print(f">> [Loader] ✅ Tìm thấy Model trên Google Drive: {model_path}")
        print(">> Đang tải từ ổ cứng (Sẽ rất nhanh)...")
        load_source = model_path
    else:
        print(f">> [Loader] ⚠️ Không tìm thấy thư mục: {model_path}")
        print(f">> [Loader] 🔄 Chuyển sang tải từ Hugging Face (Sẽ tốn thời gian)...")
        load_source = config.HF_MODEL_ID

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
                local_files_only=True if os.path.exists(model_path) else False 
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