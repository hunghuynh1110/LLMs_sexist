# config.py

# 1. Cấu hình Model
# Đường dẫn trên Drive (Nơi bạn vừa tải model về)
# LƯU Ý: Đảm bảo đường dẫn này chính xác với nơi bạn lưu trên Drive
MODEL_PATH = "/content/drive/MyDrive/Models/Llama-3.1-8B-Instruct" 

# Tên model gốc trên HF (Dùng để phòng hờ nếu không tìm thấy trên Drive)
HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 2. Cấu hình Đường dẫn file
INPUT_FILE = "prompts.json"
OUTPUT_FILE = "results.json"

# 3. Tham số chạy
TOP_K = 10
# HF_TOKEN = "..." # (Không bắt buộc nếu đã login trên Colab)