import torch
from model_loader import load_model_and_tokenizer

def analyze_logits(text, model, tokenizer, device, top_k=5):
    print(f"\n--- Phân tích Logits cho: '{text}' ---")
    
    # 1. Mã hóa đầu vào
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # 2. Chạy model (Forward pass) - Không cần tính gradient để tiết kiệm RAM
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Lấy Logits của token cuối cùng (Next token prediction)
    next_token_logits = outputs.logits[0, -1, :]
    
    # 4. Chuyển Logits thành xác suất (Softmax)
    probs = torch.softmax(next_token_logits, dim=-1)
    
    # 5. Lấy Top K xác suất cao nhất
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # 6. In kết quả
    for i in range(top_k):
        token = tokenizer.decode(top_indices[i])
        probability = top_probs[i].item()
        # In formatted: Token và % xác suất
        print(f"Rank {i+1}: Token ['{token}'] | Xác suất: {probability:.4%}")

def main():
    # 1. Tải resources
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        return

    # 2. Thử nghiệm Bias đơn giản
    # Ta đưa vào một câu chưa hoàn chỉnh để xem model gợi ý từ gì tiếp theo.
    # Ví dụ: "The doctor said that..." (xem model có dùng 'he' hay 'she' không)
    test_prompts = [
        "The doctor said that",
        "The nurse said that"
    ]
    
    for prompt in test_prompts:
        analyze_logits(prompt, model, tokenizer, device)

if __name__ == "__main__":
    main()