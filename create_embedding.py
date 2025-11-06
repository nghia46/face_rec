# create_embedding.py
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# === KHỞI TẠO INSIGHTFACE ===
print("Đang khởi tạo InsightFace (buffalo_l)...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
print("InsightFace sẵn sàng!\n")

# === CẤU HÌNH THƯ MỤC ===
INPUT_DIR = "photos"           # Thư mục chứa ảnh gốc
OUTPUT_DIR = "known_faces"     # Thư mục lưu embedding
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] Không tìm thấy thư mục: {INPUT_DIR}")
    print(f"   → Vui lòng tạo thư mục và đặt ảnh theo định dạng: NV001_Ten_Nhan_Vien.jpg")
    exit()

# === HÀM TẠO EMBEDDING ===
def create_embedding_from_image(image_path, code, name):
    """Đọc ảnh → trích xuất embedding → lưu .npy"""
    print(f"Đang xử lý: {os.path.basename(image_path)} → {code} - {name}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"   [ERROR] Không đọc được ảnh!")
        return False

    # Phát hiện khuôn mặt
    faces = app.get(img)
    if len(faces) == 0:
        print(f"   [ERROR] Không phát hiện khuôn mặt!")
        return False

    if len(faces) > 1:
        print(f"   [WARNING] Phát hiện nhiều khuôn mặt → dùng khuôn mặt đầu tiên")

    # Lấy embedding (512D, đã chuẩn hóa)
    embedding = faces[0].normed_embedding

    # Tạo tên file: NV001_Nguyen_Van_A.npy
    safe_name = name.replace(" ", "_")
    filename = f"{code}_{safe_name}.npy"
    save_path = os.path.join(OUTPUT_DIR, filename)

    np.save(save_path, embedding)
    print(f"   [SUCCESS] Đã lưu: {filename}")
    return True

# === XỬ LÝ TOÀN BỘ ẢNH TRONG THƯ MỤC ===
def batch_process():
    files = [f for f in os.listdir(INPUT_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not files:
        print(f"[ERROR] Không có ảnh nào trong {INPUT_DIR}")
        return

    success_count = 0
    print(f"Tìm thấy {len(files)} ảnh. Bắt đầu mã hóa...\n")

    for file in files:
        path = os.path.join(INPUT_DIR, file)
        # Tên file: NV001_Nguyen_Van_A.jpg → code = NV001, name = Nguyen Van A
        name_part = os.path.splitext(file)[0]
        if "_" not in name_part:
            print(f"[SKIP] Bỏ qua (tên sai định dạng): {file}")
            continue

        code, name = name_part.split("_", 1)
        name = name.replace("_", " ")  # NV001_Nguyen_Van_A → Nguyen Van A

        if create_embedding_from_image(path, code, name):
            success_count += 1

    print(f"\nHOÀN TẤT!")
    print(f"   Thành công: {success_count}/{len(files)}")
    print(f"   Đã lưu vào: {OUTPUT_DIR}/")

# === CHẠY CHƯƠNG TRÌNH ===
if __name__ == "__main__":
    batch_process()