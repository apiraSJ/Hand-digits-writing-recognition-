# Multi-Digit Handwritten Number Recognition

## เกี่ยวกับโปรเจกต์

ระบบตรวจจับและรู้จำตัวเลขลายมือหลายหลัก โดยใช้ OpenCV สำหรับ preprocessing
และ segmentation แล้วใช้ CNN ที่รับภาพขนาด 28×28 เพื่อจำแนกตัวเลข 0–9

Dashboard หลักประกอบด้วย **FastAPI backend** และ **HTML/CSS/JavaScript frontend**
รองรับการส่งภาพจากกล้องแบบ live และการอัปโหลดภาพนิ่ง ส่วน `app.py` เป็น UI
Streamlit อีกชุดหนึ่งที่แยกจากเว็บ frontend และไม่ได้ถูกเรียกใช้โดย `run_app.bat`

## โครงสร้างโปรเจกต์
```
.
├── .streamlit/
│   └── config.toml          # การตั้งค่าหน้า Streamlit
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints และเสิร์ฟ frontend
│   ├── predictor.py         # โหลดโมเดลและเรียก pipeline
│   └── segmentation.py      # ตัวช่วย segmentation สำหรับ backend
├── frontend/
│   ├── index.html           # โครงหน้า Dashboard
│   ├── style.css            # รูปแบบหน้าเว็บ
│   └── app.js               # กล้อง, ROI, เรียก API และแสดงผล
├── models/
│   └── mnist_model.keras    # โมเดล CNN ที่ backend ใช้งาน
├── pictures/                # ภาพตัวอย่าง/ภาพทดสอบ
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # grayscale, CLAHE, threshold และ MNIST canonical
│   ├── segmentation.py      # contour, รวม/แยกกล่อง และกรองตัวเลข
│   ├── recognizer.py        # โหลด Keras model และทำนาย
│   └── pipeline.py          # รวม preprocessing, segmentation, recognition
├── tests/
│   └── test_pipeline_filters.py # regression tests ของตัวกรอง
├── .gitignore               # ไฟล์และโฟลเดอร์ที่ไม่ต้องการให้ Git ติดตาม
├── app.py                   # UI Streamlit ทางเลือก (แยกจาก FastAPI)
├── Digit recognition.py     # สคริปต์รู้จำรุ่นเดิม
├── Train.csv                # label และพิกเซล 784 ค่า ต่อภาพ 28×28
├── train_model.py           # สร้าง/ฝึก CNN จาก Train.csv
├── run_app.bat              # เปิด FastAPI และ browser บน Windows
├── requirements.txt         # dependencies หลักของ backend/training
└── README.md
```

> `.venv/`, `__pycache__/` และไฟล์ชั่วคราวเป็นของเครื่อง/การรัน ไม่ใช่ source
> ที่จำเป็นต่อการส่งมอบโปรเจกต์ ไฟล์โมเดล `.keras` และโฟลเดอร์ `models/`
> ถูก ignore โดย Git; ต้องมีโมเดลในเครื่องก่อนเริ่ม API โดยเทรนด้วย `Train.csv`
> หรือเตรียมไฟล์โมเดลไว้ใน `models/mnist_model.keras`

## ลำดับการทำงาน

```text
Browser camera หรือไฟล์ภาพ
        ↓
frontend/app.js ── HTTP multipart ──> backend/main.py
                                          ↓
                                  backend/predictor.py
                                          ↓
                    src/pipeline.py → preprocessing → segmentation
                                          ↓
                              src/recognizer.py + models/
                                          ↓
                         JSON ผลทำนายกลับไปแสดงบน Dashboard
```

`backend/segmentation.py` เป็น wrapper/helper ที่ใช้ฟังก์ชันจาก `src/segmentation.py`;
logic หลักของ pipeline อยู่ใน `src/` ส่วน `app.py` เรียก pipeline โดยตรงใน UI Streamlit
ของตัวเอง

## การติดตั้ง
```bash
# 1. สร้าง virtual environment
py -3 -m venv .venv
.venv\Scripts\activate

# 2. ติดตั้ง dependencies
python -m pip install -r requirements.txt

# 3. เทรนโมเดล (ถ้ายังไม่มี models/mnist_model.keras)
python train_model.py

# 4. รันเซิร์ฟเวอร์
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

เปิดเบราว์เซอร์ที่ <http://localhost:8000>

ถ้ามีไฟล์ `models/mnist_model.keras` อยู่แล้ว สามารถข้ามขั้นตอนเทรนโมเดลได้
การเทรนใหม่จะอ่าน `Train.csv` ซึ่งต้องมี 785 คอลัมน์: label 1 คอลัมน์
และ pixel 784 คอลัมน์สำหรับภาพ 28×28

## เริ่มใช้งานด้วยคลิกเดียว (One-click startup)

ดับเบิลคลิก `run_app.bat` ได้เลย สคริปต์จะทำงานตามลำดับนี้:

1. เปลี่ยน working directory ไปยังตำแหน่งโปรเจกต์ที่กำหนดใน `PROJECT_DIR`
2. ตรวจสอบ `.venv\Scripts\python.exe`, dependencies และ `models\mnist_model.keras`
3. ถ้าแอปกำลังรันอยู่แล้ว จะ **ไม่** เปิด Uvicorn ซ้ำและ **ไม่** เปิดแท็บเบราว์เซอร์ใหม่
4. เปิด Uvicorn ที่ `127.0.0.1:8000` ในหน้าต่าง **ImageProcessing Server**
5. รอจนกว่า `GET /health` ตอบว่าพร้อมใช้งาน (ไม่ใช้แค่การหน่วงเวลาคงที่)
6. เมื่อ API พร้อมแล้ว จึงเปิด `http://127.0.0.1:8000` ในเบราว์เซอร์เริ่มต้น

หากเริ่มไม่สำเร็จ หน้าต่าง launcher และหน้าต่าง server จะค้างไว้เพื่อให้อ่านข้อผิดพลาดได้
ปิดแอปโดยปิดหน้าต่าง **ImageProcessing Server**

## การใช้งาน

หน้าเว็บหลักที่เปิดโดย `run_app.bat` เป็น Dashboard แบบ Real-time มีโหมดกล้องและอัปโหลดภาพ:

### 1. Live Camera (โหมดหลัก)
1. กด **Start Camera** แล้วอนุญาตให้เบราว์เซอร์เข้าถึงกล้อง (ใช้ได้บน localhost/HTTPS)
2. ระบบจะส่งภาพไปทำนายแบบต่อเนื่องที่อัตรา **5–10 FPS** (ค่าเริ่มต้น 6 FPS)
   โดยมี request ได้ครั้งละ 1 รายการเท่านั้น (ไม่สะสมคิวภาพ)
3. Bounding Box และป้ายตัวเลขจะถูกวาดทับบนวิดีโอโดยตรง (Overlay canvas)
4. แผงด้านขวาแสดง **Binary Threshold** แบบสด พร้อมค่า Otsu threshold และเลขเฟรม
5. กราฟด้านล่างแสดงความน่าจะเป็น 0–9 ของหลักที่เลือก
   - คลิกกรอบตัวเลขบนวิดีโอ หรือคลิกชิป `#1`, `#2` เพื่อเลือกหลัก
6. **ROI** — ค่าเริ่มต้นตรวจจับเฉพาะกรอบกลางภาพ 40% เพื่อลดการรบกวน
   เลือก `ROI: SELECT` แล้วลากกรอบเพื่อกำหนดพื้นที่เอง หรือกด Reset เพื่อกลับไปใช้กรอบกลาง
7. ปุ่ม **Stop Camera** จะหยุด inference, ยกเลิก request ที่ค้างอยู่ และคืนทรัพยากรกล้องทันที
   (รวมถึงเมื่อปิดแท็บ)

### 2. Upload Image
1. ลากไฟล์หรือคลิกเลือกภาพ JPG/PNG/BMP
2. กด **Analyze Image**
3. ระบบจะแสดงผลลัพธ์รวม, ภาพพร้อม Bounding Box, Binary Threshold,
   ภาพ 28×28 ของแต่ละหลัก และกราฟความน่าจะเป็น

### การปรับค่า
แผง **Settings** ปรับได้ทั้งสองโหมด: `min_area`, `max_area`, `merge_gap`, `padding`,
`max_digits`, `min_confidence`, `min_aspect`, `max_aspect`, inference rate และ TTA
ค่าเริ่มต้นกรองพื้นที่พิกเซลไว้ที่ 100–5000, สัดส่วนกว้าง/สูง 0.1–1.2
และความมั่นใจอย่างน้อย 80% (สามารถปรับได้ใน Settings)

### UI Streamlit ทางเลือก
`app.py` เป็น UI แยกซึ่งเรียก pipeline โดยตรง ไม่ใช่หน้าเว็บที่ FastAPI เสิร์ฟ
และไม่ได้เปิดโดย `run_app.bat`; dependencies ของ Streamlit ไม่ได้อยู่ใน
`requirements.txt` ปัจจุบัน

หมายเหตุ: กล้องจะใช้ได้เฉพาะบน `localhost`/HTTPS ตามข้อกำหนดของเบราว์เซอร์
และควรใช้ Chrome, Edge, Firefox หรือ Safari เวอร์ชันใหม่

## การทดสอบ
รัน regression tests โดยใช้ Python standard library:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## REST API
| Method | Path | รายละเอียด |
|--------|------|------------|
| `GET` | `/health` | ตรวจสอบสถานะโมเดล |
| `POST` | `/predict` | อัปโหลดภาพ (`multipart/form-data`) แล้วรับผลทำนาย |
| `POST` | `/api/predict` | เหมือน `/predict` (alias) |

พารามิเตอร์ที่รับ (Form):
`min_area`, `max_area`, `merge_gap`, `padding`, `max_digits`, `min_confidence`,
`use_tta`, `min_aspect`, `max_aspect`, `frame_id`,
`include_annotated`, `include_canonical`, `include_binary`, `include_gray`

`min_confidence` รับได้ทั้งสัดส่วน `0–1` และเปอร์เซ็นต์ `0–100`
(ค่าที่มากกว่า 1 จะถูกตีความเป็นเปอร์เซ็นต์)

ตัวอย่างการเรียก:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@digit.png" \
  -F "min_area=100" \
  -F "max_area=5000" \
  -F "merge_gap=10" \
  -F "padding=4" \
  -F "max_digits=8" \
  -F "min_confidence=0.8" \
  -F "use_tta=true" \
  -F "min_aspect=0.1" \
  -F "max_aspect=1.2"
```

ตัวอย่าง response (ตัด `binary_image`/`annotated_image` ออก):
```json
{
  "frame_id": 7,
  "server_time_ms": 1794455407368.71,
  "full_number": "1",
  "digits": [
    {
      "index": 0,
      "digit": 1,
      "confidence": 0.9905,
      "box": { "x": 30, "y": 36, "w": 15, "h": 41 },
      "probabilities": [0.0000066, 0.9905, 0.00005, 0.00005, 0.000004,
                        0.0000009, 3e-8, 0.00887, 0.00016, 0.00029],
      "canonical_image": null
    }
  ],
  "lines": ["1"],
  "annotated_image": null,
  "binary_image": "data:image/png;base64,...",
  "gray_image": null,
  "threshold": { "mode": "OTSU", "value": 163.0 },
  "image_size": { "width": 77, "height": 107 },
  "digit_count": 1,
  "average_confidence": 0.9905,
  "processing_time_ms": 286.1,
  "inference_ms": 286.1
}
```

## จุดเด่น
- Dashboard แบบ Real-time: กล้องสด + Bounding Box overlay + Binary Threshold สด
- กราฟความน่าจะเป็น 0–9 จากผลโมเดลจริง เลือกดูได้ทีละหลัก
- แสดง latency, processing FPS, confidence และประวัติการทำนาย
- ควบคุมอัตรา inference และมี request ค้างได้ครั้งละ 1 รายการ
- รองรับ ROI เพื่อจำกัดพื้นที่ตรวจจับ
- รองรับตัวเลขเดี่ยวและหลายหลัก
- ปรับตัวเองกับแสงได้ (CLAHE)
- รองรับ Test-Time Augmentation (TTA)
- แสดงความมั่นใจของแต่ละหลัก
- หน้าเว็บแยกจาก backend อย่างชัดเจน (HTML/CSS/JS + FastAPI)
- เลย์เอาต์ responsive สำหรับเดสก์ท็อปและมือถือ

## ข้อจำกัด
- ตัวเลขที่เขียนติดกันสนิท (gap 0px) อาจแยกไม่ได้
- ต้องมีช่องว่างระหว่างหลักอย่างน้อยประมาณ 1px
- โมเดลถูกเทรนจาก MNIST จึงอาจไม่แม่นกับภาพที่ต่างจากข้อมูลฝึกมาก

## ทีมงาน
โปรเจกต์นี้สร้างขึ้นเพื่อการเรียนการสอนและฝึกงานด้านคอมพิวเตอร์วิทัศน์และ Deep Learning
