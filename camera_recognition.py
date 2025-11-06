# camera_recognition.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# === KHỞI TẠO INSIGHTFACE ===
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

# === LOAD EMBEDDING TỪ known_faces ===
EMBEDDINGS_DIR = "known_faces"
KNOWN_CACHE = {}  # code -> (name, embedding)

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

if not KNOWN_CACHE:
    print("Không có nhân viên nào trong cơ sở dữ liệu!")
    exit()

# === MỞ CAMERA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

print("\nCAMERA ĐÃ MỞ – NHẤN 'q' ĐỂ THOÁT\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi đọc frame!")
        break

    # === NHẬN DIỆN KHUÔN MẶT ===
    faces = app.get(frame)
    
    for face in faces:
        # Lấy embedding
        embedding = face.normed_embedding

        # So sánh với tất cả người trong DB
        best_match = None
        best_score = 0.0
        best_code = None

        for code, (name, known_emb) in KNOWN_CACHE.items():
            score = np.dot(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_match = name
                best_code = code

        # Vẽ khung + tên
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Màu: xanh = khớp, đỏ = không
        color = (0, 255, 0) if best_score >= 0.4 else (0, 0, 255)
        label = f"{best_match} ({best_score*100:.1f}%)" if best_score >= 0.4 else "Unknown"

        # Vẽ
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === HIỂN THỊ ===
    cv2.imshow('Face Recognition - InsightFace', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === DỌN DẸP ===
cap.release()
cv2.destroyAllWindows()
print("Đã đóng camera.")