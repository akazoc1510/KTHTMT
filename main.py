import socket
import pickle
import struct
import os
import re
from datetime import datetime

import cv2
import numpy as np
import pymongo
from ultralytics import YOLO
import easyocr

# ==================================================
# CONFIG
# ==================================================
LAPTOP_IP = "192.168.1.47"
STREAM_PORT = 9999

YOLO_MODEL = "last.pt"
CONF_THRESHOLD = 0.25

OUT_DIR = "captures"
os.makedirs(OUT_DIR, exist_ok=True)

MONGO_URI = "mongodb://192.168.1.47:27017"
DB_NAME = "License_Plate_Manager"
COL_NAME = "plates"

# ==================================================
# MONGODB (KẾT NỐI 1 LẦN DUY NHẤT)
# ==================================================
mongo_client = pymongo.MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=3000,
    connectTimeoutMS=3000,
    socketTimeoutMS=3000
)
mongo_client.admin.command("ping")
print("✅ MongoDB connected")

db = mongo_client[DB_NAME]
col = db[COL_NAME]

# ==================================================
# MODELS
# ==================================================
print("⏳ Loading YOLO model...")
model = YOLO(YOLO_MODEL)

print("⏳ Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=False)

# ==================================================
# OCR HELPERS
# ==================================================
def clean_text(txt: str) -> str:
    return re.sub(r'[^A-Z0-9\-]', '', txt.upper())

def ocr_plate(crop):
    results = reader.readtext(crop, detail=1)
    best_txt, best_conf = "", 0.0

    for _, txt, conf in results:
        if conf > best_conf:
            best_txt, best_conf = txt, conf

    return clean_text(best_txt), float(best_conf)

# ==================================================
# SOCKET CONNECT
# ==================================================
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((LAPTOP_IP, STREAM_PORT))
print("📡 Connected to laptop stream")
print("👉 Nhấn 'c' TRÊN LAPTOP để chụp | 'q' để thoát")

data = b""
HEADER_SIZE = struct.calcsize(">L")
capture_idx = 0

# ==================================================
# MAIN LOOP
# ==================================================
while True:
    # --------- nhận size ----------
    while len(data) < HEADER_SIZE:
        packet = client.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < HEADER_SIZE:
        break

    packed_size = data[:HEADER_SIZE]
    data = data[HEADER_SIZE:]
    frame_size = struct.unpack(">L", packed_size)[0]

    # --------- nhận payload ----------
    while len(data) < frame_size:
        data += client.recv(4096)

    frame_data = data[:frame_size]
    data = data[frame_size:]

    # ==================================================
    # PAYLOAD TỪ LAPTOP
    # ==================================================
    payload = pickle.loads(frame_data)
    frame = payload["frame"]
    cmd = payload["cmd"]

    # ==================================================
    # CAPTURE COMMAND
    # ==================================================
    if cmd == b"CAPTURE":
        capture_idx += 1
        img_name = f"{OUT_DIR}/capture_{capture_idx}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"📸 Capture received → {img_name}")

        # ---------------- YOLO DETECT ----------------
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        plates_db = []

        if results.boxes is not None:
            for box, det_conf in zip(results.boxes.xyxy, results.boxes.conf):
                x1, y1, x2, y2 = map(int, box)

                # tránh crop lỗi
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                text, ocr_conf = ocr_plate(crop)

                plates_db.append({
                    "box": [x1, y1, x2, y2],
                    "det_conf": float(det_conf),
                    "text": text,
                    "ocr_conf": ocr_conf
                })

        # ---------------- SAVE MONGO ----------------
        col.insert_one({
            "source": "laptop_stream",
            "image": img_name,
            "created_at": datetime.utcnow(),
            "plates": plates_db
        })

        print("🗄️ Saved to MongoDB:", plates_db)

    # ==================================================
    # QUIT COMMAND
    # ==================================================
    elif cmd == b"QUIT":
        print("🛑 Quit command received")
        break

# ==================================================
# CLEANUP
# ==================================================
client.close()
print("🔌 Connection closed")
