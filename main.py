import torch
import json
import os
import time
from model_loader import load_model_and_tokenizer
from setup_data import generate_prompts

# --- IMPORT TỪ CONFIG (Thay đổi ở đây) ---
import config  # <--- Thêm dòng này

def get_top_logits(text, model, tokenizer, device, top_k=config.TOP_K): # Dùng config.TOP_K
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    next_token_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    
    # Dùng tham số từ config
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

def main():
    # Dùng config.INPUT_FILE thay vì "prompts.json"
    if not os.path.exists(config.INPUT_FILE):
        print(">> Không thấy file dữ liệu. Đang tạo mới...")
        generate_prompts(config.INPUT_FILE)
    
    with open(config.INPUT_FILE, "r") as f:
        prompts = json.load(f)
        
    # Dùng config.MODEL_ID
    model, tokenizer, device = load_model_and_tokenizer(config.MODEL_ID)
    
    print(f"\n>> Bắt đầu chạy thử nghiệm trên {len(prompts)} câu...")
    final_results = []
    start_time = time.time()
    
    for i, item in enumerate(prompts):
        text = item["text"]
        if i % 10 == 0:
            print(f"Processing {i}/{len(prompts)}: {text}") 
        
        # Truyền biến config vào hàm nếu cần
        top_tokens = get_top_logits(text, model, tokenizer, device)
        item["prediction"] = top_tokens
        final_results.append(item)
        
    # Dùng config.OUTPUT_FILE
    with open(config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n>> HOÀN TẤT! Tổng thời gian: {time.time() - start_time:.2f}s")
    print(f">> Kết quả lưu tại: {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()