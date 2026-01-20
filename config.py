import torch

# Tên mô hình (Dùng bản Instruct 8B cho nhẹ)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_device_settings():
    """
    Trả về thiết bị (device) và cấu hình quantization phù hợp.
    """
    # 1. Kiểm tra nếu là UQ Cluster (NVIDIA GPU)
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        print(">> [Config] Phát hiện: NVIDIA GPU (CUDA)")
        device = "cuda"
        # Cấu hình nén 4-bit (Chỉ dùng cho NVIDIA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        return device, bnb_config

    # 2. Kiểm tra nếu là MacBook (Apple Silicon)
    elif torch.backends.mps.is_available():
        print(">> [Config] Phát hiện: Apple Silicon (MPS)")
        device = "mps"
        # Mac không dùng bitsandbytes, trả về None
        return device, None
    
    # 3. Trường hợp CPU (Dự phòng)
    else:
        print(">> [Config] Cảnh báo: Chỉ dùng CPU")
        return "cpu", None