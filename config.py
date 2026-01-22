# config.py

# 1. Cấu hình Model
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# 2. Cấu hình Đường dẫn file
INPUT_FILE = "prompts.json"
OUTPUT_FILE = "results.json"

# 3. Tham số chạy
TOP_K = 10
HF_TOKEN = "" # (Không bắt buộc nếu đã login trên Colab)