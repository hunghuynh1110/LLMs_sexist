import torch
import json
import os
import time
from tqdm import tqdm  # Thư viện tạo thanh tiến trình
from huggingface_hub import login
from model_loader import load_model_and_tokenizer
from setup_data import generate_prompts
import config

def get_top_logits(text, model, tokenizer, device, top_k=config.TOP_K):
    """
    Hàm lấy xác suất của các token tiếp theo.
    """
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Chạy model (Inference)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Lấy logits của token cuối cùng
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Lấy top K ứng viên cao điểm nhất
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for i in range(top_k):
            token = tokenizer.decode(top_indices[i])
            results.append({
                "rank": i+1,
                "token": token,
                "prob": top_probs[i].item()
            })
        return results
    except Exception as e:
        print(f"\n[Error] Lỗi khi xử lý câu: '{text[:20]}...' | Chi tiết: {e}")
        return []

def save_intermediate(data, filename):
    """Hàm lưu file an toàn, tránh lỗi mất dữ liệu khi đang ghi"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    # 1. Đăng nhập (Dùng token từ Config hoặc biến môi trường)
    print(">> [Init] Đang kiểm tra đăng nhập...")
    try:
        # Ưu tiên lấy từ config nếu có, nếu không thì để Colab tự lo (đã login ở cell ngoài)
        if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
            login(token=config.HF_TOKEN)
    except Exception as e:
        print(f">> [Warning] Không đăng nhập qua Config được ({e}). Hy vọng bạn đã login ở ngoài.")

    # 2. Chuẩn bị dữ liệu
    if not os.path.exists(config.INPUT_FILE):
        print(">> [Data] Không thấy file input. Đang tạo mới...")
        generate_prompts(config.INPUT_FILE)
    
    with open(config.INPUT_FILE, "r") as f:
        prompts = json.load(f)

    # 3. Kiểm tra xem đã chạy được bao nhiêu rồi (Resume Capability)
    processed_count = 0
    final_results = []
    
    if os.path.exists(config.OUTPUT_FILE):
        try:
            with open(config.OUTPUT_FILE, "r") as f:
                existing_data = json.load(f)
                # Nếu file cũ hợp lệ, ta sẽ nối tiếp vào đó
                if isinstance(existing_data, list):
                    final_results = existing_data
                    processed_count = len(final_results)
                    print(f">> [Resume] Phát hiện file kết quả cũ. Đã xử lý: {processed_count} câu.")
        except:
            print(">> [Info] File kết quả cũ bị lỗi hoặc rỗng. Sẽ chạy lại từ đầu.")

    # Lọc ra những câu chưa chạy (nếu có)
    # (Ở mức đơn giản: ta giả định chạy tiếp từ index processed_count)
    prompts_to_run = prompts[processed_count:]
    
    if len(prompts_to_run) == 0:
        print(">> [Done] Tất cả dữ liệu đã được xử lý xong! Không cần chạy lại.")
        return

    # 4. Tải Model
    model, tokenizer, device = load_model_and_tokenizer(config.MODEL_PATH)    
    
    print(f"\n>> [Start] Bắt đầu xử lý {len(prompts_to_run)} câu còn lại...")
    start_time = time.time()
    
    # 5. Vòng lặp chính (Dùng tqdm để hiện thanh tiến trình)
    # desc: Mô tả, unit: đơn vị đếm
    for item in tqdm(prompts_to_run, desc="Processing", unit="prompt"):
        
        text = item["text"]
        
        # Gọi hàm tính toán
        top_tokens = get_top_logits(text, model, tokenizer, device)
        
        # Lưu kết quả vào object
        item["prediction"] = top_tokens
        final_results.append(item)
        
        # CƠ CHẾ AN TOÀN: Lưu file mỗi 10 câu (Checkpoint)
        # Để nếu sập nguồn thì chỉ mất tối đa 10 câu cuối
        if len(final_results) % 10 == 0:
            save_intermediate(final_results, config.OUTPUT_FILE)
            
    # Lưu lần cuối cùng
    save_intermediate(final_results, config.OUTPUT_FILE)
        
    elapsed = time.time() - start_time
    print(f"\n>> [Success] HOÀN TẤT! Tổng thời gian: {elapsed:.2f}s")
    print(f">> Tốc độ trung bình: {elapsed/len(prompts_to_run):.2f}s / câu")
    print(f">> Kết quả lưu tại: {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()