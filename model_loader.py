import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_ID, get_device_settings

def load_model_and_tokenizer():
    """
    Tải model và tokenizer dựa trên cấu hình thiết bị.
    """
    device, bnb_config = get_device_settings()
    
    print(f">> [Loader] Đang tải {MODEL_ID}...")
    
    # Tải Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Fix lỗi padding
    
    # Chuẩn bị tham số tải model
    load_kwargs = {
        "pretrained_model_name_or_path": MODEL_ID,
        "trust_remote_code": True,
    }
    
    # Logic riêng cho từng loại chip
    if device == "cuda":
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        # Mac chạy tốt nhất với float16
        load_kwargs["torch_dtype"] = torch.float16
        
    # Tải Model
    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        # Nếu là Mac, cần chuyển model sang device thủ công
        if device == "mps":
            model.to("mps")
            
        print(">> [Loader] Tải thành công!")
        return model, tokenizer, device
        
    except Exception as e:
        print(f">> [Loader] Lỗi nghiêm trọng: {e}")
        return None, None, None