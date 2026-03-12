import json
import re
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
Bạn là chuyên gia Ngôn ngữ học Tiếng Việt và Kỹ sư Xử lý Ngôn ngữ Tự nhiên (NLP), chuyên sâu lĩnh vực y sinh.
Nhiệm vụ: Trích xuất các CỤM TỪ CÓ Ý NGHĨA (ENTITIES) thuộc lĩnh vực y khoa, dược phẩm, lâm sàng từ thông tin sản phẩm.

════════════════════════════════════════════════════════
PHẦN 1: BA LOẠI ENTITY CẦN TRÍCH XUẤT
════════════════════════════════════════════════════════

▸ PRODUCT — Loại sản phẩm y tế, dược phẩm, chăm sóc cá nhân, thiết bị y tế:
  Ví dụ: băng gạc, que thử thai, máy đo huyết áp, thuốc nhỏ mắt, kem dưỡng da...
  • Bỏ tên thương hiệu (cả tiếng Việt lẫn nước ngoài), chỉ giữ phần mô tả loại sản phẩm.
    VD: "kem dưỡng da Sắc Ngọc Khang" → "kem dưỡng da"
        "thuốc nhỏ mắt Sancoba"       → "thuốc nhỏ mắt"
  • Dạng bào chế (dung dịch, gel...) chỉ lấy khi đi kèm đường dùng cụ thể.
    VD: "dung dịch nhỏ mắt" ✅ — "dung dịch" đứng một mình ❌

▸ ANATOMY — Bộ phận, cơ quan, hệ thống cơ thể người:
  Ví dụ: mắt, mũi, da, bắp tay, đường hô hấp, mạch máu, niêm mạc...
  • Bao gồm bộ phận đứng một mình (mắt, mũi...) lẫn cụm ghép (xoang mũi, niêm mạc mắt...).
  • KHÔNG lấy danh từ chỉ nhóm người: phụ nữ, trẻ em, bệnh nhân...

▸ PATHOLOGY — Bệnh lý hoặc triệu chứng lâm sàng được công nhận (có tên gọi lâm sàng / mã ICD):
  Ví dụ: viêm xoang, nghẹt mũi, ung thư, nám, tàn nhang, ưu trương...
  • CHỈ lấy DANH TỪ / CỤM DANH TỪ chỉ tên bệnh hoặc triệu chứng lâm sàng rõ ràng.
  • KHÔNG lấy:
    - Tính từ / mô tả thẩm mỹ mơ hồ: sạm, lão hóa, mỏi mắt, khô da...
    - Tiền tố "chứng" + tính từ: "chứng mỏi mắt" → bỏ hoàn toàn
    - Kết quả xét nghiệm: dương tính, âm tính
    - Cụm thời gian / giai đoạn: sau sinh, sau mổ...

════════════════════════════════════════════════════════
PHẦN 2: BLACKLIST — LOẠI BỎ HOÀN TOÀN
════════════════════════════════════════════════════════

- Tên thương hiệu, từ tiếng Anh, hoạt chất ngoại lai: Sancoba, natri, ibuprofen, vitamin...
- Danh từ chỉ nhóm người: phụ nữ, trẻ em, người cao tuổi...
- Tính từ / mô tả mơ hồ: sạm, lão hóa, dương tính, kháng khuẩn, siêu mỏng...
- Dạng bào chế đứng độc lập: "dung dịch", "viên nén"...
- Cụm thời gian: sau sinh, sau mổ, trước phẫu thuật...
- Từ chung phi chuyên ngành: y tế, thiên nhiên, giải pháp, bề mặt...
- Động từ / thao tác: bảo quản, làm sạch, cải thiện, ngăn ngừa...
- Địa điểm / tên công ty: bệnh viện, Nhật Bản, Công ty...
- Cụm chứa số, ký tự đặc biệt, ký tự không phải tiếng Việt.

════════════════════════════════════════════════════════
PHẦN 3: QUY TẮC CHUNKING
════════════════════════════════════════════════════════

- Entity phải là DANH TỪ hoặc CỤM DANH TỪ, độ dài 1–5 âm tiết.
- KHÔNG cắt vụn khái niệm đã trọn vẹn:
  ✅ "máy đo huyết áp", "kính áp tròng", "viêm đường hô hấp"
  ❌ "máy đo", "áp tròng", "đường hô hấp" tách khỏi "viêm"
- Không lặp entity trùng nhau.
- Không suy đoán, không thêm thông tin ngoài văn bản.

════════════════════════════════════════════════════════
PHẦN 4: VÍ DỤ MẪU
════════════════════════════════════════════════════════

Input:
  Kem dưỡng da ban đêm Sắc Ngọc Khang làm mờ nám, sạm, tàn nhang. Ngăn ngừa lão hóa.
  Thuốc nhỏ mắt Sancoba cải thiện chứng mỏi mắt. Máy đo huyết áp bắp tay Panasonic.
  Nước muối sinh lý ưu trương xịt mũi cho người bị nghẹt mũi, viêm xoang.

Output:
[
  {"term": "kem dưỡng da",      "type": "PRODUCT"},
  {"term": "nám",               "type": "PATHOLOGY"},
  {"term": "tàn nhang",         "type": "PATHOLOGY"},
  {"term": "da",                "type": "ANATOMY"},
  {"term": "thuốc nhỏ mắt",    "type": "PRODUCT"},
  {"term": "mắt",               "type": "ANATOMY"},
  {"term": "máy đo huyết áp",  "type": "PRODUCT"},
  {"term": "bắp tay",           "type": "ANATOMY"},
  {"term": "nước muối sinh lý", "type": "PRODUCT"},
  {"term": "ưu trương",         "type": "PATHOLOGY"},
  {"term": "mũi",               "type": "ANATOMY"},
  {"term": "nghẹt mũi",         "type": "PATHOLOGY"},
  {"term": "viêm xoang",        "type": "PATHOLOGY"}
]

════════════════════════════════════════════════════════
OUTPUT
════════════════════════════════════════════════════════

- Chuyển toàn bộ về chữ thường (lowercase).
- Trả về DUY NHẤT một JSON array. KHÔNG giải thích, KHÔNG markdown.
- Mỗi phần tử gồm đúng hai key: "term" và "type" (PRODUCT | ANATOMY | PATHOLOGY).

[{"term": "<exact text>", "type": "<TYPE>"}, ...]

Nếu không có entity hợp lệ → trả về: []
""".strip()

# =============================================================================
# EXTRACTION
# =============================================================================

def extract_medical_entities(text: str) -> list:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\nVăn bản cần phân tích:\n{text}",
        config=genai.types.GenerateContentConfig(temperature=0),
    )
    raw   = response.text or "[]"
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Medical NER API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


@app.post("/api/extract")
async def extract(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY chưa được cấu hình trong .env")
    try:
        entities = extract_medical_entities(body.text)
        print(f"Extracted entities: {entities}")
        return {"entities": entities}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model trả về JSON không hợp lệ")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   GEMINI_MODEL,
        "ready":   bool(GEMINI_API_KEY),
    }


# Static frontend — mount cuối cùng
app.mount("/", StaticFiles(directory="static", html=True), name="static")
