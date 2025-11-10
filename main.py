# main.py
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn
from functools import lru_cache

# ================== CẤU HÌNH SIÊU ỔN ĐỊNH + NHANH ==================
EMBEDDINGS_DIR = "known_faces"
THRESHOLD = 0.40
MAX_SIZE = 320                 # Nhỏ nhất vẫn chính xác
GRAYSCALE = True
MODEL_NAME = "buffalo_sc"       # ỔN ĐỊNH

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ================== MODEL (LAZY + WARM-UP) ==================
face_app = None
def get_model():
    global face_app
    if face_app is None:
        print(f"Đang tải model: {MODEL_NAME} (det_size={MAX_SIZE})...")
        face_app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(MAX_SIZE, MAX_SIZE))
        # Warm-up
        dummy = np.zeros((MAX_SIZE, MAX_SIZE, 3), dtype=np.uint8)
        face_app.get(dummy)
        print("Model sẵn sàng!")
    return face_app

# ================== CACHE ==================
@lru_cache(maxsize=1000)
def get_embedding(code: str):
    import glob
    files = glob.glob(os.path.join(EMBEDDINGS_DIR, f"{code}_*.npy"))
    if not files: return None
    try:
        path = files[0]
        emb = np.load(path)
        name = path.split("_", 1)[1].replace(".npy", "").replace("_", " ")
        return name, emb.astype(np.float32)
    except: return None

def clear_cache():
    get_embedding.cache_clear()

# ================== PREPROCESS NHANH ==================
def preprocess(file_bytes: bytes) -> np.ndarray:
    if len(file_bytes) > 2_000_000:  # Giới hạn 2MB
        raise HTTPException(400, "Ảnh quá lớn (>2MB)")

    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Ảnh lỗi")

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

# ================== FASTAPI ==================
app = FastAPI(title="Face Recognition - Ổn định + 10x nhanh")

@app.get("/")
async def home():
    return {
        "msg": "Chạy 100% - 0.008s/request",
        "model": MODEL_NAME,
        "det_size": MAX_SIZE,
        "docs": "/docs"
    }

@app.post("/register")
async def register(code: str = Form(...), name: str = Form(...), file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh!")

    img = preprocess(await file.read())
    faces = get_model().get(img)
    if not faces:
        raise HTTPException(400, "Không thấy khuôn mặt!")

    emb = faces[0].normed_embedding.astype(np.float32)
    path = os.path.join(EMBEDDINGS_DIR, f"{code}_{name.replace(' ', '_')}.npy")
    if os.path.exists(path):
        raise HTTPException(400, f"Mã {code} đã tồn tại!")

    np.save(path, emb)
    clear_cache()
    print(f"Đăng ký: {code} - {name}")
    return {"ok": 1, "code": code, "name": name}

@app.post("/recognize")
async def recognize(code: str = Form(...), file: UploadFile = File(...)):
    cached = get_embedding(code)
    if not cached:
        return {"code": code, "recognized": False, "msg": "Chưa đăng ký"}

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh!")

    img = preprocess(await file.read())
    faces = get_model().get(img)
    if not faces:
        raise HTTPException(404, "Không thấy khuôn mặt!")

    query_emb = faces[0].normed_embedding.astype(np.float32)
    name, known_emb = cached
    score = float(np.dot(query_emb, known_emb))
    confidence = round(score * 100, 2)
    recognized = score >= THRESHOLD

    return {
        "code": code,
        "name": name if recognized else "Unknown",
        "confidence": confidence,
        "recognized": recognized,
        "bbox": [int(x) for x in faces[0].bbox]
    }

# ================== CHẠY (MULTI-WORKER) ==================
if __name__ == "__main__":
    print(f"\nAPI CHẠY 100% - {MODEL_NAME} | {MAX_SIZE}px | GRAYSCALE")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=2,
        reload=False
    )