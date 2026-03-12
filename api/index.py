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
Nhiệm vụ: Trích xuất các CỤM TỪ CÓ Ý NGHĨA (ENTITIES) thuộc lĩnh vực y khoa, dược phẩm, lâm sàng từ thông tin sản phẩm. Độ dài linh hoạt dựa trên ranh giới khái niệm độc lập (thường 2–5 âm tiết).

════════════════════════════════════════════════════════
PHẦN 1: QUY TẮC PHÂN LOẠI TỪ VỰNG (DOMAIN FILTERING)
════════════════════════════════════════════════════════

【1.1】 CHỈ GIỮ LẠI (TRÍCH XUẤT) CÁC NHÓM SAU — MỖI ENTITY PHẢI THUỘC ĐÚNG MỘT TRONG BA LOẠI:

▸ PRODUCT — Tên sản phẩm y tế, dược phẩm, chăm sóc cá nhân, thiết bị y tế, dược mỹ phẩm:
  Ví dụ: băng gạc, nhiệt kế, sữa rửa mặt, băng vệ sinh, bao cao su, que thử thai,
          máy đo huyết áp, kính áp tròng, khẩu trang, xịt mũi, gel bôi...
  ⚠️ Dạng bào chế (dung dịch, gel, viên nén...) CHỈ được trích xuất khi đứng cùng
     với tên sản phẩm hoặc chỉ định rõ ràng (vd: "dung dịch xịt mũi").
     KHÔNG trích xuất dạng bào chế đứng độc lập, chung chung (vd: chỉ "dung dịch").

▸ ANATOMY — Bộ phận, cơ quan, hệ thống giải phẫu cơ thể người:
  Ví dụ: hô hấp, bắp tay, âm đạo, âm hộ, dương vật, não, mạch máu, da, niêm mạc,
          đường hô hấp, xoang mũi...
  ⚠️ CHỈ lấy danh từ chỉ BỘ PHẬN / CƠ QUAN cơ thể.
     KHÔNG lấy danh từ chỉ nhóm người / đối tượng (phụ nữ, trẻ em, người cao tuổi...).

▸ PATHOLOGY — Bệnh lý hoặc triệu chứng lâm sàng cụ thể:
  Ví dụ: viêm xoang, cảm lạnh, đau đầu, viêm gan B, ung thư, nghẹt mũi, sổ mũi,
          phong hàn, đột quỵ, rung nhĩ, ưu trương...
  ⚠️ CHỈ lấy danh từ / cụm danh từ chỉ BỆNH hoặc TRIỆU CHỨNG có thể chẩn đoán
     hoặc quan sát lâm sàng.
     KHÔNG lấy:
     - Tính từ / trạng thái kết quả xét nghiệm đứng độc lập: dương tính, âm tính
     - Cụm trạng từ chỉ thời gian / giai đoạn: sau sinh, trước phẫu thuật, sau mổ
     - Trạng thái sinh lý thông thường không mang tính bệnh lý: mang thai, hành kinh

【1.2】 ❌ LOẠI BỎ HOÀN TOÀN — BLACKLIST:

  (a) Ngôn ngữ & ký tự:
      - Từ tiếng Anh, phiên âm quốc tế, hoạt chất ngoại lai:
        natri, hyaluronate, ibuprofen, vitamin, omega, collagen, panasonic, covid...
      - Cụm chứa số (0–9), ký tự đặc biệt, dấu câu, ký tự không phải tiếng Việt.

  (b) Danh từ chỉ đối tượng / nhóm người dùng:
      phụ nữ, trẻ em, người lớn, người cao tuổi, bệnh nhân, thai phụ...

  (c) Tính từ & trạng thái kết quả — không phải thực thể y khoa độc lập:
      dương tính, âm tính, siêu mỏng, kháng khuẩn, tự động, sinh lý...

  (d) Cụm trạng từ chỉ thời gian / giai đoạn:
      sau sinh, trước sinh, sau mổ, sau phẫu thuật...

  (e) Dạng bào chế / đường dùng đứng độc lập quá chung:
      dung dịch (đứng một mình), viên nén, bột...

  (f) Từ mô tả chung, hành chính, phi chuyên ngành:
      y tế, tự động, thiên nhiên, kết hợp, giải pháp, bề mặt, môi trường...

  (g) Động từ / hành động thao tác:
      bảo quản, làm sạch, sử dụng, bơm xả, cố định, làm loãng...

  (h) Địa điểm / cơ sở:
      bệnh viện, phòng khám, nha khoa, thẩm mỹ...

════════════════════════════════════════════════════════
PHẦN 2: QUY TẮC PHÂN TÁCH CỤM TỪ (CHUNKING)
════════════════════════════════════════════════════════

- TRÍCH XUẤT TRỌN VẸN cụm từ là một khái niệm độc lập (1–5 âm tiết).
- KHÔNG chia nhỏ danh từ ghép đã có nghĩa trọn vẹn:
  ✅ ĐÚNG: "kính áp tròng", "máy đo huyết áp", "viêm đường hô hấp", "que thử thai"
  ❌ SAI : "kính áp", "máy đo", "đường hô hấp" tách khỏi "viêm" (cắt vụn khái niệm)
- Entity phải là DANH TỪ hoặc CỤM DANH TỪ — không phải tính từ, động từ, trạng từ.
- Không lặp lại entity trùng nhau.
- Không suy đoán, không thêm thông tin ngoài văn bản đầu vào.
- Không sửa chính tả, không loại bỏ dấu tiếng Việt.

════════════════════════════════════════════════════════
PHẦN 3: VÍ DỤ MẪU (FEW-SHOT LEARNING)
════════════════════════════════════════════════════════

Input:
  Máy đo huyết áp bắp tay tự động Panasonic EW3109.
  Nước muối sinh lý ưu trương xịt mũi dành cho người bị nghẹt mũi, viêm xoang.
  Dung dịch ngâm bảo quản kính áp tròng.
  Băng vệ sinh siêu mỏng dành cho phụ nữ sau sinh.
  Que thử thai dương tính.
  Khẩu trang y tế kháng khuẩn bảo vệ đường hô hấp.

Output:
[
  {"term": "máy đo huyết áp",  "type": "PRODUCT"},
  {"term": "bắp tay",           "type": "ANATOMY"},
  {"term": "nước muối sinh lý", "type": "PRODUCT"},
  {"term": "ưu trương",         "type": "PATHOLOGY"},
  {"term": "nghẹt mũi",         "type": "PATHOLOGY"},
  {"term": "viêm xoang",        "type": "PATHOLOGY"},
  {"term": "kính áp tròng",     "type": "PRODUCT"},
  {"term": "băng vệ sinh",      "type": "PRODUCT"},
  {"term": "que thử thai",      "type": "PRODUCT"},
  {"term": "khẩu trang",        "type": "PRODUCT"},
  {"term": "đường hô hấp",      "type": "ANATOMY"}
]

════════════════════════════════════════════════════════
YÊU CẦU ĐẦU RA (OUTPUT)
════════════════════════════════════════════════════════
- Chuyển toàn bộ về chữ thường (lowercase).
- Trả về DUY NHẤT một JSON array hợp lệ. KHÔNG giải thích, KHÔNG dùng markdown code fence.
- Mỗi phần tử gồm đúng hai key:
    "term" : cụm từ trích xuất (chữ thường, đúng nguyên văn trong văn bản)
    "type" : một trong ba giá trị → PRODUCT | ANATOMY | PATHOLOGY

Định dạng bắt buộc:
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
