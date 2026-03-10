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
ROLE
You are a Vietnamese medical NER system specialized in pharmaceutical product descriptions.

TASK
Extract unique medical entity phrases from Vietnamese pharmaceutical text (500–1000 words).
Return ONLY a valid JSON array. No explanation. No duplicate terms.

ENTITY TYPES
- ACTIVE_INGREDIENT : Chemical or pharmaceutical substance with therapeutic effect. Min 1 word. Include well-known names like "vitamin c", "omega-3" even if they contain hyphens or letters.
- DISEASE           : Diagnosed disease or pathological condition. Min 2 words.
- SYMPTOM           : Abnormal sensation or sign perceived by patient. Min 1 word.
- BODY_PART         : Body part related to disease or treatment. Min 1 word.
- DOSAGE_FORM       : Physical form of a drug (tablet, capsule, syrup...). Min 2 words.
- MEDICAL_DEVICE    : Medical equipment or consumable for diagnosis/care. Min 2 words.
- HEALTHCARE_PRODUCT: Non-drug health or cosmetic product for body care. Min 2 words.
- MEDICAL_TERM      : Specialized medical terminology not covered above. Min 2 words.
- PATIENT_CONDITION : Patient health status relevant to disease or treatment. Min 2 words.

RULES: 
1. Vietnamese only. Skip any phrase that mixes Vietnamese and English inseparably. If separating yields a valid Vietnamese entity, extract only the Vietnamese part.
2. Extract exact substrings from the source text. No normalization, no spelling correction, no inference.
3. Skip abbreviations (e.g. "HA", "TB").
4. Skip phrases containing digits EXCEPT well-known substance names (e.g. "vitamin c", "omega-3", "b12").
5. Skip generic drug/medicine words (e.g. "thuốc", "viên uống") unless they form a meaningful medical phrase.
6. If a phrase qualifies for multiple types, emit one entry per matching type.
7. If a phrase qualifies for both DISEASE and SYMPTOM, emit both.
8. Deduplicate: each unique term appears only once per type.

OUTPUT FORMAT
[{"term": "<exact text>", "type": "<TYPE>"}]
Return [] if no valid entities found.
"""

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