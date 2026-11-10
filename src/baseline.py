"""
src/baseline.py – Phương pháp đối chứng không dùng LLM (Rule-based Baseline)
=============================================================================

TẠI SAO CẦN PHƯƠNG PHÁP ĐỐI CHỨNG? (Theo đề bài & Bài 11):
─────────────────────────────────────────────────────────────
1. Đề bài yêu cầu: Phải có phương pháp đối chứng không dùng LLM và so sánh
   trên ít nhất 100 nhãn tay (accuracy/F1), kèm phân tích lỗi.
2. Để chứng minh việc bỏ tiền/tài nguyên chạy LLM có mang lại giá trị vượt trội hay không.
   Nếu một bộ lọc từ khoá đơn giản chạy trong 0.01 giây đã đạt 80% F1 thì có đáng
   bỏ nghìn USD gọi LLM? LLM chỉ chứng minh được giá trị khi vượt qua baseline ở các
   câu có ngữ cảnh phức tạp, hàm ý mỉa mai, đa ngôn ngữ, hoặc câu chứa cả khen lẫn chê (mixed).

CƠ CHẾ HOẠT ĐỘNG:
─────────────────
- Sử dụng Từ điển cảm xúc (Lexicon-based) và Từ điển khía cạnh (Aspect keywords).
- Sentiment:
  + positive: số từ tích cực > số từ tiêu cực (và không có từ phủ định mạnh)
  + negative: số từ tiêu cực > số từ tích cực
  + mixed: có cả khen và chê rõ rệt
- Aspect extraction: Kiểm tra sự xuất hiện của các từ khoá đại diện trong câu.
- Language detection: Nhận diện cơ bản các ngôn ngữ phổ biến (en, es, fr, zh, vi, und).
"""

import re
from typing import Dict, List, Any


# Từ điển từ vựng cảm xúc tiếng Anh & thông dụng
POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "perfect",
    "lovely", "clean", "spotless", "spacious", "comfortable", "friendly",
    "helpful", "convenient", "quiet", "beautiful", "superb", "enjoyed",
    "recommend", "best", "nice", "loved", "good", "happy", "cozy", "peaceful"
}

NEGATIVE_WORDS = {
    "dirty", "noisy", "loud", "terrible", "horrible", "awful", "bad",
    "rude", "broken", "uncomfortable", "disappointed", "poor", "smell",
    "dusty", "stain", "cold", "hot", "bug", "cockroach", "refund", "never",
    "worst", "scam", "late", "cancelled", "cancel", "unresponsive", "unsafe"
}

ASPECT_KEYWORDS = {
    "vị trí": [
        "location", "metro", "train", "station", "bus", "walk", "walkable",
        "close to", "near", "beach", "harbour", "cbd", "central", "distance",
        "conveniently located", "ferry", "bridge"
    ],
    "sạch sẽ": [
        "clean", "cleanliness", "spotless", "dirty", "dust", "dusty", "smell",
        "stain", "tidy", "hygiene", "cockroach", "bugs", "insects"
    ],
    "chủ nhà": [
        "host", "owner", "communication", "responsive", "friendly", "helpful",
        "rude", "check-in", "check in", "unresponsive", "hospitality", "welcome"
    ],
    "giá trị": [
        "price", "value", "expensive", "cheap", "worth", "money", "cost",
        "affordable", "overpriced", "bargain"
    ],
    "tiện nghi": [
        "wifi", "internet", "bed", "mattress", "shower", "bathroom", "kitchen",
        "air conditioning", "ac", "aircon", "heating", "pool", "balcony", "towel",
        "linen", "amenities", "appliances", "tv"
    ],
    "độ ồn": [
        "noise", "noisy", "quiet", "loud", "soundproof", "traffic", "street noise",
        "construction", "barking", "peaceful"
    ],
    "an toàn": [
        "safe", "safety", "secure", "lock", "security", "dark", "sketchy",
        "dangerous", "police"
    ],
}


def _tokenize(text: str) -> List[str]:
    """Tách từ đơn giản bằng regex theo chữ cái thường."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def detect_language_simple(text: str) -> str:
    """Nhận diện ngôn ngữ sơ bộ cho baseline."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "und"
    
    # Kiểm tra ký tự đặc trưng
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", text, re.IGNORECASE):
        return "vi"
    if re.search(r"[éàèùâêîôûëïüç]", text, re.IGNORECASE):
        return "fr"
    if re.search(r"[áéíóúñ¿¡]", text, re.IGNORECASE):
        return "es"
    
    # Mặc định văn bản chữ la-tinh là tiếng Anh trong tập dữ liệu Sydney
    return "en"


def classify_review_baseline(text: str) -> Dict[str, Any]:
    """Phân loại cảm xúc và khía cạnh bằng luật/từ điển.
    
    Returns:
        dict với schema tương đương LLM:
        {
            "sentiment": "positive" | "negative" | "mixed",
            "aspects_positive": [...],
            "aspects_negative": [...],
            "language": "en" | ...
        }
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "sentiment": "positive",
            "aspects_positive": [],
            "aspects_negative": [],
            "language": "und"
        }

    text_lower = text.lower()
    tokens = set(_tokenize(text))

    pos_count = len(tokens.intersection(POSITIVE_WORDS))
    neg_count = len(tokens.intersection(NEGATIVE_WORDS))

    # Quyết định cảm xúc
    if pos_count > 0 and neg_count > 0:
        sentiment = "mixed"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        # Trong Airbnb, đa phần review không có từ tiêu cực đều mang tính tích cực
        sentiment = "positive"

    # Trích xuất khía cạnh (aspects)
    aspects_found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            aspects_found.append(aspect)

    # Gán hướng tích cực / tiêu cực cho khía cạnh
    aspects_pos = []
    aspects_neg = []
    
    if sentiment == "positive":
        aspects_pos = aspects_found
    elif sentiment == "negative":
        aspects_neg = aspects_found
    else:
        # mixed: chia đôi hoặc gán ngữ cảnh gần (ở baseline gán đều cho cả 2 để đơn giản)
        aspects_pos = [a for a in aspects_found if a in ["vị trí", "chủ nhà", "tiện nghi"]]
        aspects_neg = [a for a in aspects_found if a in ["sạch sẽ", "độ ồn", "giá trị", "an toàn"]]
        if not aspects_pos and not aspects_neg:
            aspects_pos = aspects_found

    return {
        "sentiment": sentiment,
        "aspects_positive": aspects_pos,
        "aspects_negative": aspects_neg,
        "language": detect_language_simple(text)
    }


def predict_batch_baseline(reviews_df: Any, text_col: str = "comments") -> List[Dict[str, Any]]:
    """Chạy đối chứng trên toàn bộ DataFrame đánh giá."""
    results = []
    for _, row in reviews_df.iterrows():
        comment = row.get(text_col, "")
        pred = classify_review_baseline(str(comment))
        pred["review_id"] = row.get("id", None)
        results.append(pred)
    return results
