import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

# === KHỞI TẠO INSIGHTFACE ===
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# === LOAD EMBEDDINGS ===
EMBEDDINGS_DIR = "known_faces"
KNOWN_CACHE = {}

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_DIR):
        print(f"[ERROR] Thư mục {EMBEDDINGS_DIR} không tồn tại!")
        return
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            code = file.split("_", 1)[0]
            path = os.path.join(EMBEDDINGS_DIR, file)
            try:
                emb = np.load(path)
                name = file.split("_", 1)[1].replace(".npy", "").replace("_", " ")
                KNOWN_CACHE[code] = (name, emb)
                print(f"[LOADED] {code} - {name}")
            except Exception as e:
                print(f"[ERROR] {file}: {e}")

load_embeddings()

# === FASTAPI APP ===
app = FastAPI(title="Face Recognition API - 404 on No Face")

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    # Kiểm tra loại file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh!")

    # Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Không thể đọc ảnh. File không hợp lệ.")

    # Phát hiện khuôn mặt
    faces = face_app.get(frame)

    # === KHÔNG CÓ KHUÔN MẶT → 404 ===
    if not faces:
        raise HTTPException(
            status_code=404,
            detail="Không phát hiện khuôn mặt nào trong ảnh."
        )

    # === CÓ KHUÔN MẶT → XỬ LÝ NHẬN DIỆN ===
    results = []
    for face in faces:
        embedding = face.normed_embedding
        best_score = 0.0
        best_name = "Unknown"
        best_code = None

        for code, (name, known_emb) in KNOWN_CACHE.items():
            score = np.dot(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_name = name
                best_code = code

        confidence = float(best_score * 100)
        recognized = bool(best_score >= 0.4)
        bbox = [int(float(x)) for x in face.bbox]

        results.append({
            "code": best_code,
            "name": best_name if recognized else "Unknown",
            "confidence": round(confidence, 2),
            "bbox": bbox,
            "recognized": recognized
        })

    return {"results": results}

# === CHẠY SERVER ===
if __name__ == "__main__":
    print("\nAPI RUNNING: http://127.0.0.1:8000")
    print("→ Không có khuôn mặt → HTTP 404")
    uvicorn.run(app, host="127.0.0.1", port=8000)