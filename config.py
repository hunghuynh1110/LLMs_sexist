# config.py

# 1. Cấu hình Model
# Khi nào chạy thật trên Cluster thì đổi thành bản 70B ở đây
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

# 2. Cấu hình Đường dẫn file
INPUT_FILE = "prompts.json"
OUTPUT_FILE = "results.json"

# 3. Cấu hình tham số chạy
TOP_K = 10      # Lấy top 10 token có xác suất cao nhất
DEVICE_MAP = "auto"

# 4. Token (Nếu muốn để ở đây thay vì login thủ công - Tùy chọn)
# HF_TOKEN = "hf_xxxx..."