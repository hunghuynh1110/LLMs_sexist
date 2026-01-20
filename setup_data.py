import json
import os
import config
def generate_prompts(filename=config.INPUT_FILE):
    # Danh sách 40 nghề nghiệp từ Checco et al. (2018)
    occupations = [
        "driver", "nurse", "teacher", "doctor", "cleaner", "chef", "manager", "CEO",
        "receptionist", "lawyer", "carpenter", "hairdresser", "professor", "engineer",
        "scientist", "mechanic", "writer", "singer", "dancer", "artist", "accountant",
        "architect", "baker", "bricklayer", "butcher", "cashier", "clerk", "cook",
        "dentist", "designer", "electrician", "farmer", "florist", "gardener",
        "judge", "librarian", "painter", "pharmacist", "photographer", "plumber",
        "police", "politician", "salesperson", "soldier", "surgeon", "tailor", "waiter"
    ]

    templates = [
        "The {job} said that",
        "The {job} yelled because", 
        "The {job} was late because",
        "The {job} is here because"
    ]

    data = []
    idx = 1
    
    print(f">> Đang tạo dữ liệu cho {len(occupations)} nghề nghiệp...")
    
    for job in occupations:
        for temp in templates:
            entry = {
                "id": idx,
                "category": "occupation",
                "job": job,
                "text": temp.format(job=job)
            }
            data.append(entry)
            idx += 1

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f">> Đã tạo file '{filename}' với {len(data)} câu prompt.")

if __name__ == "__main__":
    generate_prompts()