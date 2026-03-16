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

- PRODUCT — Loại sản phẩm y tế, dược phẩm, thực phẩm chức năng, thực phẩm bổ sung sức khỏe,
  chăm sóc cá nhân, thiết bị y tế, tên bài thuốc / thuốc đông y cổ truyền:
  Ví dụ: băng gạc, que thử thai, máy đo huyết áp, thuốc nhỏ mắt...
  • Bỏ tên thương hiệu thương mại (Sancoba, Bioslim...), chỉ giữ phần mô tả loại sản phẩm.
    VD: "kem dưỡng da Sắc Ngọc Khang" → "kem dưỡng da"
  • GIỮ LẠI tên bài thuốc đông y / Hán-Việt có nghĩa lâm sàng, ghép với loại sản phẩm.
  • Dạng bào chế chỉ lấy khi đi kèm đường dùng: "dung dịch nhỏ mắt" ✅ — "dung dịch" ❌

- ANATOMY — Bộ phận, cơ quan, hệ thống cơ thể người:
  Ví dụ: mắt, da, xoang mũi, niêm mạc, đường hô hấp...

- PATHOLOGY — Bệnh lý hoặc triệu chứng lâm sàng được công nhận:
  Ví dụ: viêm xoang, nghẹt mũi, táo bón, nám, tàn nhang...

════════════════════════════════════════════════════════
PHẦN 2: BLACKLIST — LOẠI BỎ HOÀN TOÀN
════════════════════════════════════════════════════════

- Tên thương hiệu thương mại, từ tiếng Anh, hoạt chất ngoại lai: Sancoba, Bioslim, natri, ibuprofen...
- Danh từ chỉ nhóm người: phụ nữ, trẻ em, người cao tuổi...
- Tính từ / mô tả mơ hồ: sạm, lão hóa, kháng khuẩn, siêu mỏng...
- Dạng bào chế đứng độc lập: "dung dịch", "viên nén", "viên uống","gói"...
- Cụm thời gian, động từ, địa điểm, tên công ty, từ chung phi chuyên ngành.
- Cụm chứa số hoặc ký tự đặc biệt.

════════════════════════════════════════════════════════
PHẦN 3: QUY TẮC CHUNKING
════════════════════════════════════════════════════════

- Entity phải là DANH TỪ hoặc CỤM DANH TỪ, không cắt vụn khái niệm trọn vẹn.
- Không lặp entity trùng nhau. Không suy đoán ngoài văn bản.
- Chỉ extract CỤM TỪ LIỀN KỀ trong văn bản gốc. KHÔNG ghép, KHÔNG tổng hợp các từ không đứng cạnh nhau.

════════════════════════════════════════════════════════
PHẦN 4: VÍ DỤ MẪU
════════════════════════════════════════════════════════


Input:  Kem dưỡng da ban đêm Sắc Ngọc Khang làm mờ nám, tàn nhang. Ngăn ngừa lão hóa.
Output:
{"entities": [{"term": "kem dưỡng da", "type": "PRODUCT"}, {"term": "nám", "type": "PATHOLOGY"}, {"term": "tàn nhang", "type": "PATHOLOGY"}, {"term": "da", "type": "ANATOMY"}]}

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════

- Toàn bộ "term" viết thường (lowercase).
- Trả về DUY NHẤT một JSON object: {"entities": [...]}
- Mỗi phần tử gồm đúng hai key: "term" và "type" (PRODUCT | ANATOMY | PATHOLOGY).
- KHÔNG giải thích, KHÔNG markdown code fence.
- Nếu không có entity hợp lệ → {"entities": []}
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
