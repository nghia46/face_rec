import cv2
import numpy as np
import os
import glob
from insightface.app import FaceAnalysis
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import ORJSONResponse
import uvicorn
from typing import Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image

# ================== CẤU HÌNH ==================
EMBEDDINGS_DIR = "known_faces"
THRESHOLD = 0.50              # Nhận diện: 50% trở lên
DUPLICATE_THRESHOLD = 0.65    # Trùng lặp: 65% trở lên (cao hơn 15%)
MAX_SIZE = 256
MAX_FILE_SIZE = 2_000_000
MODEL_NAME = "buffalo_sc"

# Preprocessing configs
USE_PILLOW = True
SKIP_GRAYSCALE = True
TARGET_FORMAT = "RGB"

# Thread pool
executor = ThreadPoolExecutor(max_workers=4)

# Messages
MSG_NOT_REGISTERED = "Chưa đăng ký"
MSG_UNKNOWN = "Unknown"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ================== VALIDATION ==================
def validate_thresholds():
    """Kiểm tra logic ngưỡng"""
    if DUPLICATE_THRESHOLD <= THRESHOLD:
        raise ValueError(
            f"LOGIC SAI!\n"
            f"DUPLICATE_THRESHOLD ({DUPLICATE_THRESHOLD}) phải CAO HƠN THRESHOLD ({THRESHOLD})\n"
            f"Lý do: Nếu không, người đủ điểm nhận diện sẽ bị chặn khi đăng ký!\n"
            f"Khuyến nghị: DUPLICATE = THRESHOLD + 0.10 đến 0.15"
        )
    gap = DUPLICATE_THRESHOLD - THRESHOLD
    if gap < 0.05:
        print(f"Cảnh báo: Khoảng cách giữa 2 ngưỡng quá nhỏ ({gap:.2f})")
        print(f"    Khuyến nghị: Tối thiểu 0.10 để tránh xung đột")
validate_thresholds()

# ================== HELPER FUNCTIONS ==================
def extract_name_from_path(path: str) -> str:
    """Trích xuất tên từ đường dẫn file"""
    try:
        filename = os.path.basename(path)
        name_part = filename.split("_", 1)[1].replace(".npy", "")
        return name_part.replace("_", " ")
    except (IndexError, AttributeError):
        return MSG_UNKNOWN

def find_embedding_file(code: str) -> Optional[str]:
    """Tìm file embedding theo code"""
    files = glob.glob(os.path.join(EMBEDDINGS_DIR, f"{code}_*.npy"))
    return files[0] if files else None

# ================== MODEL ==================
face_app = None
def get_model() -> FaceAnalysis:
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

# ================== CACHE (SỬA LỖI) ==================
embeddings_cache = {}

def preload_all_embeddings():
    """Load tất cả embeddings vào RAM – ĐẢM BẢO ĐỒNG BỘ"""
    global embeddings_cache
    embeddings_cache.clear()
    files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npy"))
    loaded = 0
    for path in files:
        try:
            filename = os.path.basename(path)
            code = filename.split("_", 1)[0]
            emb = np.load(path).astype(np.float32)
            name = extract_name_from_path(path)
            embeddings_cache[code] = (name, emb)
            loaded += 1
        except Exception as e:
            print(f"[ERROR] Lỗi load {path}: {e}")
    print(f"Đã load {loaded}/{len(files)} embeddings vào cache")

def get_embedding(code: str) -> Optional[Tuple[str, np.ndarray]]:
    """Lấy embedding trực tiếp từ cache dict"""
    return embeddings_cache.get(code)

def refresh_cache():
    """Reload cache sau mỗi thay đổi (register/delete)"""
    preload_all_embeddings()

# ================== DUPLICATE DETECTION ==================
def check_duplicate_face(new_embedding: np.ndarray, exclude_code: str = None) -> Optional[Tuple[str, str, float]]:
    max_similarity = 0.0
    duplicate_code = None
    duplicate_name = None
    
    for code, (name, known_emb) in embeddings_cache.items():
        if code == exclude_code:
            continue
        similarity = float(np.dot(new_embedding, known_emb))
        if similarity > max_similarity:
            max_similarity = similarity
            duplicate_code = code
            duplicate_name = name
    
    if max_similarity >= DUPLICATE_THRESHOLD:
        return (duplicate_code, duplicate_name, max_similarity)
    return None

async def check_duplicate_face_async(new_embedding: np.ndarray, exclude_code: str = None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, check_duplicate_face, new_embedding, exclude_code)

# ================== PREPROCESSING ==================
def preprocess_pillow(file_bytes: bytes) -> np.ndarray:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"Ảnh quá lớn (>{MAX_FILE_SIZE // 1_000_000}MB)")
    try:
        img_pil = Image.open(BytesIO(file_bytes))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        w, h = img_pil.size
        if max(w, h) > MAX_SIZE:
            img_pil.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
        img_np = np.asarray(img_pil, dtype=np.uint8)
        if not img_np.flags['C_CONTIGUOUS']:
            img_np = np.ascontiguousarray(img_np)
        return img_np
    except Exception as e:
        raise HTTPException(400, f"Ảnh lỗi: {str(e)}")

def preprocess_opencv(file_bytes: bytes) -> np.ndarray:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"Ảnh quá lớn (>{MAX_FILE_SIZE // 1_000_000}MB)")
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Ảnh lỗi")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img

def preprocess_sync(file_bytes: bytes) -> np.ndarray:
    return preprocess_pillow(file_bytes) if USE_PILLOW else preprocess_opencv(file_bytes)

async def preprocess(file_bytes: bytes) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, preprocess_sync, file_bytes)

# ================== FACE DETECTION ==================
def detect_faces_sync(img: np.ndarray):
    return get_model().get(img)

async def detect_faces(img: np.ndarray):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, detect_faces_sync, img)

# ================== CONFIDENCE CLASSIFICATION ==================
def classify_confidence(score: float) -> dict:
    if score >= 0.70:
        return {"level": "CERTAIN", "label": "Chắc chắn", "color": "#22c55e", "emoji": "Chắc chắn"}
    elif score >= 0.60:
        return {"level": "VERY_LIKELY", "label": "Rất có khả năng", "color": "#84cc16", "emoji": "Rất có khả năng"}
    elif score >= 0.50:
        return {"level": "LIKELY", "label": "Có khả năng", "color": "#eab308", "emoji": "Có khả năng"}
    elif score >= 0.40:
        return {"level": "MAYBE", "label": "Nghi ngờ", "color": "#f97316", "emoji": "Nghi ngờ"}
    else:
        return {"level": "UNLIKELY", "label": "Không phải", "color": "#ef4444", "emoji": "Không phải"}

# ================== FASTAPI ==================
app = FastAPI(
    title="Face Recognition API (Ultra Fast + Duplicate Check)",
    default_response_class=ORJSONResponse
)

@app.on_event("startup")
async def startup_event():
    print("Đang khởi động...")
    get_model()
    preload_all_embeddings()
    print(f"Sẵn sàng! (Preprocessing: {'PIL' if USE_PILLOW else 'OpenCV'})")
    print(f"Recognition threshold: {THRESHOLD} ({THRESHOLD*100:.0f}%)")
    print(f"Duplicate threshold: {DUPLICATE_THRESHOLD} ({DUPLICATE_THRESHOLD*100:.0f}%)")
    print(f"Gap: {DUPLICATE_THRESHOLD - THRESHOLD:.2f}\n")

@app.get("/")
async def home():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "det_size": MAX_SIZE,
        "threshold": THRESHOLD,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "threshold_gap": round(DUPLICATE_THRESHOLD - THRESHOLD, 2),
        "preprocessing": "PIL" if USE_PILLOW else "OpenCV",
        "cached_faces": len(embeddings_cache),
        "docs": "/docs"
    }

@app.post("/register")
async def register(
    code: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh (image/*)")
    
    if code in embeddings_cache:
        old_name = embeddings_cache[code][0]
        raise HTTPException(400, f"Mã {code} đã được đăng ký cho: {old_name}")
    
    file_bytes = await file.read()
    img = await preprocess(file_bytes)
    faces = await detect_faces(img)
    
    if not faces:
        raise HTTPException(400, "Không phát hiện khuôn mặt trong ảnh")
    
    new_embedding = faces[0].normed_embedding.astype(np.float32)
    duplicate = await check_duplicate_face_async(new_embedding)
    if duplicate:
        dup_code, dup_name, similarity = duplicate
        raise HTTPException(
            409,
            f"Khuôn mặt đã được đăng ký!\nMã: {dup_code} | Tên: {dup_name}\nĐộ tương đồng: {similarity*100:.1f}%"
        )
    
    safe_name = name.replace(' ', '_')
    path = os.path.join(EMBEDDINGS_DIR, f"{code}_{safe_name}.npy")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, np.save, path, new_embedding)
    
    embeddings_cache[code] = (name, new_embedding)
    refresh_cache()  # ← ĐẢM BẢO ĐỒNG BỘ
    
    print(f"Đăng ký: {code} - {name}")
    return {
        "success": True,
        "code": code,
        "name": name,
        "message": "Đăng ký thành công"
    }

@app.post("/recognize")
async def recognize(
    code: str = Form(...),
    file: UploadFile = File(...)
):
    cached = get_embedding(code)
    if not cached:
        return {
            "code": code,
            "recognized": False,
            "message": MSG_NOT_REGISTERED
        }
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh (image/*)")
    
    file_bytes = await file.read()
    img = await preprocess(file_bytes)
    faces = await detect_faces(img)
    
    if not faces:
        raise HTTPException(404, "Không phát hiện khuôn mặt trong ảnh")
    
    query_emb = faces[0].normed_embedding.astype(np.float32)
    name, known_emb = cached
    score = float(np.dot(query_emb, known_emb))
    confidence = round(score * 100, 2)
    recognized = score >= THRESHOLD
    conf_class = classify_confidence(score)
    
    return {
        "code": code,
        "name": name if recognized else MSG_UNKNOWN,
        "confidence": confidence,
        "confidence_level": conf_class["level"],
        "confidence_label": conf_class["label"],
        "recognized": recognized,
        "bbox": [int(x) for x in faces[0].bbox]
    }

@app.post("/check-duplicate")
async def check_duplicate_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh (image/*)")
    
    file_bytes = await file.read()
    img = await preprocess(file_bytes)
    faces = await detect_faces(img)
    
    if not faces:
        raise HTTPException(400, "Không phát hiện khuôn mặt trong ảnh")
    
    new_embedding = faces[0].normed_embedding.astype(np.float32)
    duplicate = await check_duplicate_face_async(new_embedding)
    
    if duplicate:
        dup_code, dup_name, similarity = duplicate
        return {
            "is_duplicate": True,
            "matched_code": dup_code,
            "matched_name": dup_name,
            "similarity": round(similarity * 100, 2)
        }
    else:
        return {
            "is_duplicate": False,
            "message": "Khuôn mặt chưa được đăng ký"
        }

@app.delete("/delete/{code}")
async def delete_by_code(code: str):
    if code not in embeddings_cache:
        raise HTTPException(404, f"Không tìm thấy mã {code}")
    
    name, _ = embeddings_cache[code]
    file_path = find_embedding_file(code)
    if file_path and os.path.exists(file_path):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, os.remove, file_path)
    
    del embeddings_cache[code]
    refresh_cache()  # ← Đồng bộ
    
    print(f"Đã xóa: {code} - {name}")
    return {
        "success": True,
        "deleted_code": code,
        "deleted_name": name,
        "message": f"Đã xóa {name} (mã: {code})"
    }

@app.post("/delete-by-name")
async def delete_by_name(name: str = Form(...)):
    deleted = []
    name_lower = name.lower()
    codes_to_delete = [code for code, (cached_name, _) in embeddings_cache.items() if cached_name.lower() == name_lower]
    
    if not codes_to_delete:
        raise HTTPException(404, f"Không tìm thấy người có tên '{name}'")
    
    for code in codes_to_delete:
        cached_name, _ = embeddings_cache[code]
        file_path = find_embedding_file(code)
        if file_path and os.path.exists(file_path):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, os.remove, file_path)
        del embeddings_cache[code]
        deleted.append({"code": code, "name": cached_name})
        print(f"Đã xóa: {code} - {cached_name}")
    
    refresh_cache()
    return {
        "success": True,
        "deleted_count": len(deleted),
        "deleted_records": deleted,
        "message": f"Đã xóa {len(deleted)} bản ghi có tên '{name}'"
    }

@app.delete("/delete-all")
async def delete_all(confirm: str = Form(...)):
    if confirm != "DELETE_ALL":
        raise HTTPException(400, "Xác nhận không đúng! Nhập 'DELETE_ALL' để xóa tất cả")
    
    if not embeddings_cache:
        return {"success": True, "deleted_count": 0, "message": "Không có dữ liệu để xóa"}
    
    deleted_count = len(embeddings_cache)
    files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npy"))
    loop = asyncio.get_event_loop()
    for file_path in files:
        await loop.run_in_executor(executor, os.remove, file_path)
    
    embeddings_cache.clear()
    refresh_cache()
    
    print(f"ĐÃ XÓA TẤT CẢ: {deleted_count} khuôn mặt")
    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"Đã xóa toàn bộ {deleted_count} khuôn mặt"
    }

@app.get("/list")
async def list_all():
    if not embeddings_cache:
        return {"total": 0, "faces": [], "message": "Chưa có khuôn mặt nào được đăng ký"}
    
    faces = []
    for code, (name, _) in embeddings_cache.items():
        file_path = find_embedding_file(code)
        file_size = os.path.getsize(file_path) if file_path else 0
        faces.append({
            "code": code,
            "name": name,
            "file_size_kb": round(file_size / 1024, 2)
        })
    faces.sort(key=lambda x: x["code"])
    return {"total": len(faces), "faces": faces}

@app.get("/stats")
async def stats():
    return {
        "total_registered": len(embeddings_cache),
        "preprocessing": "PIL (LANCZOS)" if USE_PILLOW else "OpenCV (LINEAR)",
        "max_size": MAX_SIZE,
        "threshold": THRESHOLD,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "threshold_gap": round(DUPLICATE_THRESHOLD - THRESHOLD, 2),
        "logic_valid": DUPLICATE_THRESHOLD > THRESHOLD
    }

@app.get("/test-thresholds")
async def test_thresholds():
    test_scores = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
    results = []
    for score in test_scores:
        conf_class = classify_confidence(score)
        results.append({
            "similarity": score,
            "confidence_pct": round(score * 100, 1),
            "would_recognize": score >= THRESHOLD,
            "would_block_duplicate": score >= DUPLICATE_THRESHOLD,
            "classification": conf_class["label"],
            "emoji": conf_class["emoji"]
        })
    return {
        "threshold_config": {
            "recognition": THRESHOLD,
            "duplicate": DUPLICATE_THRESHOLD,
            "gap": round(DUPLICATE_THRESHOLD - THRESHOLD, 2)
        },
        "test_results": results,
        "explanation": {
            "recognition_zone": f"≥ {THRESHOLD:.2f} → Nhận diện thành công",
            "gray_zone": f"{THRESHOLD:.2f} - {DUPLICATE_THRESHOLD:.2f} → Nhận diện OK, có thể đăng ký người khác",
            "duplicate_zone": f"≥ {DUPLICATE_THRESHOLD:.2f} → Chặn đăng ký (trùng lặp)"
        }
    }

# ================== RUN ==================
if __name__ == "__main__":
    print(f"\nFace Recognition API (ULTRA FAST + DUPLICATE CHECK)")
    print(f"Model: {MODEL_NAME} | Size: {MAX_SIZE}px")
    print(f"Preprocessing: {'PIL (3x faster)' if USE_PILLOW else 'OpenCV'}")
    print(f"Recognition threshold: {THRESHOLD} ({THRESHOLD*100:.0f}%)")
    print(f"Duplicate threshold: {DUPLICATE_THRESHOLD} ({DUPLICATE_THRESHOLD*100:.0f}%)")
    print(f"Gap: {DUPLICATE_THRESHOLD - THRESHOLD:.2f}\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=2,
        reload=False,
        log_level="warning"
    )