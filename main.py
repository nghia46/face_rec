# main.py
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn
from functools import lru_cache

# ================== CẤU HÌNH ==================
EMBEDDINGS_DIR = "known_faces"
THRESHOLD = 0.40
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ================== MODEL (LAZY INIT) ==================
face_app = None

def get_model():
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return face_app

# ================== CACHE + XÓA TỰ ĐỘNG ==================
@lru_cache(maxsize=500)
def get_embedding(code: str):
    import glob
    files = glob.glob(os.path.join(EMBEDDINGS_DIR, f"{code}_*.npy"))
    if not files:
        return None
    try:
        path = files[0]
        emb = np.load(path)
        name = path.split("_", 1)[1].replace(".npy", "").replace("_", " ")
        return name, emb.astype(np.float32)
    except:
        return None

# Hàm xóa cache khi đăng ký mới
def clear_cache_for_code(code: str):
    get_embedding.cache_clear()  # XÓA TOÀN BỘ CACHE (an toàn, nhanh)
    # Hoặc xóa riêng: get_embedding.cache.pop(code, None)

# ================== FASTAPI ==================
app = FastAPI(title="Face Recognition 1:1 - REALTIME")

@app.get("/")
async def home():
    return {"message": "API nhận diện realtime - không cần restart!", "docs": "/docs"}

# ================== ĐĂNG KÝ + XÓA CACHE ==================
@app.post("/register")
async def register(code: str = Form(...), name: str = Form(...), file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh!")

    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Ảnh không hợp lệ.")

    faces = get_model().get(img)
    if not faces:
        raise HTTPException(400, "Không tìm thấy khuôn mặt!")

    emb = faces[0].normed_embedding.astype(np.float32)
    path = os.path.join(EMBEDDINGS_DIR, f"{code}_{name.replace(' ', '_')}.npy")

    if os.path.exists(path):
        raise HTTPException(400, f"Mã {code} đã tồn tại!")

    np.save(path, emb)
    clear_cache_for_code(code)  # XÓA CACHE → REALTIME
    print(f"Đăng ký mới: {code} - {name}")
    return {"message": "Đăng ký thành công", "code": code, "name": name}

# ================== NHẬN DIỆN 1:1 (REALTIME) ==================
@app.post("/recognize")
async def recognize(code: str = Form(...), file: UploadFile = File(...)):
    # Kiểm tra code trước (realtime)
    if not get_embedding(code):
        return {"code": code, "recognized": False, "message": "Không phải nhân viên đã đăng ký"}

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh!")

    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Ảnh không hợp lệ.")

    faces = get_model().get(img)
    if not faces:
        raise HTTPException(404, "Không tìm thấy khuôn mặt!")

    query_emb = faces[0].normed_embedding.astype(np.float32)
    name, known_emb = get_embedding(code)

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

# ================== CHẠY ==================
if __name__ == "__main__":
    print("http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)