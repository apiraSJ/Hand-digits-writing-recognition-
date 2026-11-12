"""
app.py
------
Streamlit Web App สำหรับ Multi-Digit Handwritten Number Recognition
รองรับ 2 โหมด:
  1. Image & Snapshot Mode  – อัปโหลดภาพ หรือถ่ายภาพจากกล้อง
  2. Webcam Real-time Mode  – สตรีมวิดีโอสดแบบ Real-time (ประมวลผลทีละเฟรม)

หน้าแรกเป็น Landing Page แบบการดำ 2 ใบ (First Menu) แล้วจึงเข้าสู่โหมดที่เลือก
"""

import io
import os
import time
from typing import TypedDict

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import DigitPredictor, PipelineResult, run_pipeline
from src.recognizer import DigitRecognizer

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_model.keras")

BAR_COLORS = [
    "#6366f1", "#f97316", "#22c55e", "#ef4444", "#a855f7",
    "#eab308", "#ec4899", "#64748b", "#14b8a6", "#0ea5e9",
]

MODE_MENU = "menu"
MODE_IMAGE = "image"
MODE_WEBCAM = "webcam"


class PipelineKwargs(TypedDict):
    min_area: int
    max_area: int
    merge_gap: int
    padding: int
    max_digits: int
    min_confidence: float
    use_tta: bool
    min_aspect: float
    max_aspect: float


# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Digit Handwritten Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 55%, #22d3ee 100%);
        border-radius: 26px;
        padding: 46px 44px 40px;
        color: #ffffff;
        box-shadow: 0 18px 40px -18px rgba(79, 70, 229, .65);
        margin-bottom: 28px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, .18);
        border: 1px solid rgba(255, 255, 255, .35);
        border-radius: 999px;
        padding: 5px 14px;
        font-size: .78rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .hero h1 {
        font-size: 2.45rem;
        line-height: 1.18;
        font-weight: 800;
        margin: 0 0 14px;
        color: #ffffff;
    }
    .hero p {
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 680px;
        margin: 0;
        color: #e8f6ff;
    }
    .hero-stats {
        display: flex;
        gap: 34px;
        margin-top: 26px;
        flex-wrap: wrap;
    }
    .hero-stat b { display: block; font-size: 1.45rem; font-weight: 800; }
    .hero-stat span { font-size: .8rem; color: #d7efff; }

    /* ---------- Mode cards ---------- */
    .mode-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 26px 26px 20px;
        height: 100%;
        min-height: 250px;
        box-shadow: 0 10px 30px -20px rgba(15, 23, 42, .45);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .mode-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 36px -20px rgba(15, 23, 42, .5);
    }
    .mode-icon {
        font-size: 2.4rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 62px; height: 62px;
        border-radius: 18px;
        background: #eef2ff;
        margin-bottom: 14px;
    }
    .mode-card h3 { margin: 0 0 8px; font-size: 1.22rem; color: #0f172a; }
    .mode-card p  { margin: 0 0 14px; font-size: .9rem; color: #64748b; line-height: 1.55; }
    .chip {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        border-radius: 999px;
        padding: 4px 11px;
        font-size: .74rem;
        margin: 0 6px 6px 0;
    }

    /* ---------- Result ---------- */
    .result-box {
        background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
        border-radius: 20px;
        padding: 22px 30px;
        text-align: center;
        margin: 6px 0 22px;
        box-shadow: 0 16px 34px -22px rgba(79, 70, 229, .8);
    }
    .result-number {
        font-size: 3.6rem;
        font-weight: 900;
        color: #fff;
        letter-spacing: .18em;
        line-height: 1.1;
    }
    .result-label { font-size: .92rem; color: #dcf1ff; margin-top: 4px; }

    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0f172a;
        margin: 20px 0 10px;
    }

    /* ---------- Digit tiles ---------- */
    .digit-tile {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px 10px;
        text-align: center;
        box-shadow: 0 8px 20px -18px rgba(15, 23, 42, .5);
    }
    .digit-tile .digit { font-size: 1.9rem; font-weight: 800; color: #4f46e5; line-height: 1.1; }
    .digit-tile .conf  { font-size: .78rem; color: #64748b; margin-top: 2px; }
    .conf-good { color: #16a34a; font-weight: 700; }
    .conf-mid  { color: #d97706; font-weight: 700; }
    .conf-low  { color: #dc2626; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  Helper: โหลดโมเดล (cache)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ กำลังโหลดโมเดล CNN...")
def load_recognizer() -> DigitRecognizer:
    return DigitRecognizer(MODEL_PATH)


# ─────────────────────────────────────────────
#  Helper: แปลงภาพ
# ─────────────────────────────────────────────
def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def gray_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def decode_image(data: bytes) -> np.ndarray | None:
    """แปลง bytes ของไฟล์ภาพ → ภาพ BGR (คืน None ถ้าอ่านไม่ได้)"""
    if not data:
        return None
    buffer = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Helper: คำนวณ confidence class
# ─────────────────────────────────────────────
def confidence_class(conf: float) -> str:
    if conf >= 0.90:
        return "conf-good"
    if conf >= 0.70:
        return "conf-mid"
    return "conf-low"


# ─────────────────────────────────────────────
#  Helper: โปรบาทความน่าจะเป็น
# ─────────────────────────────────────────────
def prob_chart_html(probs: np.ndarray, title: str = "") -> str:
    best = int(np.argmax(probs))
    bars = ""
    for i, p in enumerate(probs):
        pct = float(p) * 100
        weight = "font-weight:800;" if i == best else ""
        bars += f"""
        <div style="display:flex;align-items:center;gap:7px;margin:3px 0;">
          <span style="width:16px;text-align:right;font-size:.78rem;color:#334155;{weight}">{i}</span>
          <div style="flex:1;background:#eef2f7;border-radius:5px;height:13px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{BAR_COLORS[i]};border-radius:5px;height:13px;"></div>
          </div>
          <span style="width:44px;font-size:.72rem;color:#64748b;">{pct:.1f}%</span>
        </div>"""
    return f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px;">
      <p style="font-size:.78rem;font-weight:700;margin:0 0 6px;color:#0f172a;">{title}</p>
      {bars}
    </div>"""


# ─────────────────────────────────────────────
#  Helper: แสดงผลลัพธ์จาก Pipeline
# ─────────────────────────────────────────────
def render_pipeline_result(result: PipelineResult) -> None:
    if not result.digits:
        st.warning(
            "⚠️ ไม่พบตัวเลขในภาพ — ลองเขียนตัวเลขให้ใหญ่ขึ้น "
            "หรือปรับพารามิเตอร์ Segmentation ในแถบด้านข้าง"
        )
        return

    avg_conf = float(np.mean([d.confidence for d in result.digits])) * 100
    st.markdown(
        f"""
        <div class="result-box">
          <div class="result-number">{result.full_number}</div>
          <div class="result-label">
            รู้จำได้ {len(result.digits)} หลัก · ความมั่นใจเฉลี่ย {avg_conf:.1f}%
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col_ann, col_bin = st.columns(2, gap="medium")
    with col_ann:
        st.markdown('<p class="section-title">📸 ภาพพร้อม Bounding Box</p>', unsafe_allow_html=True)
        st.image(bgr_to_pil(result.annotated_image), width="stretch")
    with col_bin:
        st.markdown('<p class="section-title">🔲 ภาพ Binary Threshold</p>', unsafe_allow_html=True)
        st.image(gray_to_pil(result.binary_image), width="stretch")

    st.markdown(
        '<p class="section-title">🧮 ขั้นตอนการคิด (Calculation Breakdown)</p>',
        unsafe_allow_html=True,
    )
    st.caption("ภาพ 28×28 ด้านล่างคือ 'สิ่งที่โมเดลเห็นจริง' หลังผ่านการแปลงแบบ MNIST")

    per_row = min(len(result.digits), 5)
    columns = st.columns(per_row, gap="medium")
    for i, digit_result in enumerate(result.digits):
        column = columns[i % per_row]
        with column:
            st.image(
                gray_to_pil(digit_result.canonical_28),
                caption=f"หลักที่ {i + 1}",
                width=96,
            )
            conf_pct = digit_result.confidence * 100
            st.markdown(
                f"""
                <div class="digit-tile">
                  <div class="digit">{digit_result.digit}</div>
                  <div class="conf">ความมั่นใจ
                    <span class="{confidence_class(digit_result.confidence)}">{conf_pct:.1f}%</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander("ดูความน่าจะเป็น 0–9"):
                st.markdown(
                    prob_chart_html(
                        digit_result.probabilities,
                        f"หลักที่ {i + 1} — ทำนาย {digit_result.digit}",
                    ),
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
#  Session state & กล้อง
# ─────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = MODE_MENU
if "cam_on" not in st.session_state:
    st.session_state.cam_on = False


def release_camera() -> None:
    """ปิดกล้องและคืนทรัพยากร (เรียกได้ทุกเมื่อที่ออกจากโหมด Webcam)"""
    camera = st.session_state.get("cam")
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    st.session_state.cam = None
    st.session_state.cam_index = None


def go_to(mode: str) -> None:
    if mode != MODE_WEBCAM:
        release_camera()
        st.session_state["force_stop_cam"] = True
    st.session_state.mode = mode
    st.rerun()


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ การตั้งค่า")
    st.caption("พารามิเตอร์การแบ่งตัวเลขและรู้จำ")

    st.markdown("#### 🔧 Segmentation")
    min_area = st.slider(
        "พื้นที่ขั้นต่ำของตัวเลข (px²)", 50, 1000, 100, 10,
        help="กรองจุดเล็ก ๆ / สัญญาณรบกวนออก",
    )
    max_area = st.slider(
        "พื้นที่สูงสุดของตัวเลข (px²)", 1000, 20000, 5000, 500,
        help="กรองเงาหรือวัตถุขนาดใหญ่ออกจากการตรวจจับ",
    )
    merge_gap = st.slider(
        "ระยะห่างการรวมชิ้นส่วน (px)", 0, 30, 10, 1,
        help="ระยะแนวตั้งสูงสุดที่ถือว่าเป็นชิ้นส่วนของเลขหลักเดียวกัน",
    )
    padding = st.slider(
        "Padding รอบตัวเลข (px)", 0, 20, 4, 1,
        help="เพิ่มขอบรอบตัวเลขก่อนแปลงเป็นรูปแบบ MNIST",
    )

    st.markdown("#### 🎯 การรู้จำ")
    max_digits = st.slider("จำนวนหลักสูงสุด", 1, 10, 8, 1)
    min_confidence = st.slider(
        "ความมั่นใจขั้นต่ำ (%)", 0, 95, 80, 5,
        help="ทิ้งผลทำนายที่มั่นใจน้อยกว่านี้ (0 = แสดงทุกหลัก)",
    )
    min_aspect = st.slider(
        "สัดส่วนกว้าง/สูงขั้นต่ำ", 0.05, 0.5, 0.1, 0.05,
    )
    max_aspect = st.slider(
        "สัดส่วนกว้าง/สูงสูงสุด", 0.5, 3.0, 1.2, 0.1,
    )
    use_tta = st.toggle(
        "Test-Time Augmentation", value=True,
        help="เพิ่มความแม่นยำโดยดูภาพ 2 มุมมอง (ช้าลงเล็กน้อย)",
    )

    st.markdown("---")
    if os.path.isfile(MODEL_PATH):
        st.success("✅ โมเดลพร้อมใช้งาน")
        st.caption("`models/mnist_model.keras`")
    else:
        st.error("❌ ไม่พบไฟล์โมเดล")
        st.caption("รัน `python train_model.py` ก่อน")

    st.markdown("---")
    if st.session_state.mode != MODE_MENU:
        if st.button("🏠 กลับหน้าแรก", width="stretch"):
            go_to(MODE_MENU)


PIPELINE_KWARGS: PipelineKwargs = dict(
    min_area=min_area,
    max_area=max_area,
    merge_gap=merge_gap,
    padding=padding,
    max_digits=max_digits,
    min_confidence=min_confidence / 100.0,
    use_tta=use_tta,
    min_aspect=min_aspect,
    max_aspect=max_aspect,
)


def _run_with_settings(
    image: np.ndarray,
    recognizer: DigitPredictor,
) -> PipelineResult:
    return run_pipeline(
        image,
        recognizer,
        min_area=PIPELINE_KWARGS["min_area"],
        max_area=PIPELINE_KWARGS["max_area"],
        merge_gap=PIPELINE_KWARGS["merge_gap"],
        padding=PIPELINE_KWARGS["padding"],
        max_digits=PIPELINE_KWARGS["max_digits"],
        min_confidence=PIPELINE_KWARGS["min_confidence"],
        use_tta=PIPELINE_KWARGS["use_tta"],
        min_aspect=PIPELINE_KWARGS["min_aspect"],
        max_aspect=PIPELINE_KWARGS["max_aspect"],
    )


# ─────────────────────────────────────────────
#  โหลดโมเดล
# ─────────────────────────────────────────────
try:
    recognizer = load_recognizer()
except FileNotFoundError:
    recognizer = None
except Exception as error:  # pragma: no cover - กรณีโมเดลเสียหาย
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {error}")
    st.stop()


# ═══════════════════════════════════════════════════════
#  หนาแรก (First Menu)
# ═══════════════════════════════════════════════════════
def render_menu() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-badge">CNN · OpenCV · MNIST</div>
          <h1>Multi-Digit Handwritten<br>Number Recognition</h1>
          <p>
            ระบบรู้จำตัวเลขลายมือหลายหลัก (เช่น 542) ด้วยโครงข่ายประสาทเทียม CNN
            และการประมวลผลภาพด้วย OpenCV — เลือกโหมดที่ต้องการด้านล่างเพื่อเริ่มต้น
          </p>
          <div class="hero-stats">
            <div class="hero-stat"><b>28×28</b><span>ขนาดภาพเข้าโมเดล</span></div>
            <div class="hero-stat"><b>0–9</b><span>คลาสที่รู้จำได้</span></div>
            <div class="hero-stat"><b>หลายหลัก</b><span>รองรับการแบ่งตัวเลข</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recognizer is None:
        st.error(
            f"❌ ไม่พบไฟล์โมเดล: `{MODEL_PATH}`\n\n"
            "กรุณารัน `python train_model.py` ก่อนใช้งาน"
        )
        st.stop()

    col_image, col_webcam = st.columns(2, gap="large")

    with col_image:
        st.markdown(
            """
            <div class="mode-card">
              <div class="mode-icon">🖼️</div>
              <h3>Image & Snapshot</h3>
              <p>อัปโหลดภาพถ่ายหรือภาพวาดลายมือ แล้วระบบจะแบ่งตัวเลขแต่ละหลัก
                 พร้อมแสดงขั้นตอนการคิดทีละขั้น</p>
              <div>
                <span class="chip">อัปโหลดไฟล์</span>
                <span class="chip">ถ่ายจากกล้อง</span>
                <span class="chip">Probability Chart</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "เลือกโหมดภาพนิ่ง →", key="btn_mode_image",
            type="primary", width="stretch",
        ):
            go_to(MODE_IMAGE)

    with col_webcam:
        st.markdown(
            """
            <div class="mode-card">
              <div class="mode-icon">🎥</div>
              <h3>Webcam Real-time</h3>
              <p>สตรีมภาพจากเว็บแคมและรู้จำตัวเลขแบบสด ๆ ทีละเฟรม
                 เหมาะสำหรับการเดโมหน้าชั้นเรียน</p>
              <div>
                <span class="chip">Real-time</span>
                <span class="chip">กรอบโฟกัสกลางจอ</span>
                <span class="chip">เริ่ม/หยุดได้</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "เลือกโหมดเว็บแคม →", key="btn_mode_webcam",
            type="primary", width="stretch",
        ):
            go_to(MODE_WEBCAM)

    st.markdown("")
    st.info(
        "💡 เคล็ดลับ: เขียนตัวเลขให้ชัด ใช้ปากกาสีเข้มบนกระดาษสีขาว "
        "และให้ตัวเลขอยู่ในกรอบขาวตรงกลาง จะได้ผลลัพธ์แม่นที่สุด"
    )

    # --- แสดงตัวอย่างภาพตัวอย่าง ---
    with st.expander("📸 ตัวอย่างภาพที่ระบบรองรับ", expanded=False):
        st.markdown("**ภาพเดี่ยว:** แสดงเลข 0–9 เป็นรายตัว")
        sample_cols = st.columns(5)
        sample_info = [
            ("0", "ศูนย์"),
            ("1", "หนึ่ง"),
            ("2", "สอง"),
            ("3", "สาม"),
            ("4", "สี่"),
        ]
        for i, (label, desc) in enumerate(sample_info):
            with sample_cols[i]:
                st.write(f"**{label}**: {desc}")

        st.markdown("**ภาพหลายหลัก:** แสดงเลข 542 (ระยะห่างปานกลาง)")
        st.caption("โดยภาพจะถูกประมวลผลผ่านกรอบโฟกัสกลางจอ")

        st.markdown("**ขั้นตอนการประมวลผล:**")
        st.markdown("1. ตัด ROI กลางจอ → 2. Threshold + กลับสีอัตโนมัติตามสีพื้นหลัง → 3. หา Contour → 4. มาย์ CN เป็นรูปแบบ MNIST → 5. ทำนายผล")

    st.markdown("")


# ═══════════════════════════════════════════════════════
#  โหมด 1: Image & Snapshot
# ═══════════════════════════════════════════════════════
def render_image_mode() -> None:
    st.markdown("## 🖼️ Image & Snapshot Mode")
    st.caption("อัปโหลดภาพหรือถ่ายภาพจากกล้อง แล้วระบบจะแบ่งและรู้จำตัวเลขให้ทันที")

    tab_upload, tab_camera = st.tabs(["📂 อัปโหลดภาพ", "📸 ถ่ายภาพจากกล้อง"])

    with tab_upload:
        uploaded = st.file_uploader(
            "เลือกไฟล์ภาพ (JPG / PNG / BMP)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="รองรับภาพที่มีตัวเลขหนึ่งหลักหรือหลายหลัก",
        )
        if uploaded is not None:
            image = decode_image(uploaded.read())
            if image is None:
                st.error("❌ อ่านไฟล์ภาพไม่ได้ — ลองใช้ไฟล์ JPG หรือ PNG")
            else:
                with st.spinner("🔍 กำลังวิเคราะห์..."):
                    result = _run_with_settings(image, recognizer)
                render_pipeline_result(result)

    with tab_camera:
        st.info("📸 กดปุ่มด้านล่างเพื่อถ่ายภาพตัวเลขด้วยเว็บแคม")
        snapshot = st.camera_input("ถ่ายภาพตัวเลข")
        if snapshot is not None:
            image = decode_image(snapshot.read())
            if image is None:
                st.error("❌ อ่านภาพจากกล้องไม่ได้ — ลองถ่ายใหม่")
            else:
                with st.spinner("🔍 กำลังวิเคราะห์..."):
                    result = _run_with_settings(image, recognizer)
                render_pipeline_result(result)


# ═══════════════════════════════════════════════════════
#  โหมด 2: Webcam Real-time (per-frame rerun)
# ═══════════════════════════════════════════════════════
def render_webcam_mode() -> None:
    st.markdown("## 🎥 Webcam Real-time Mode")
    st.caption(
        "ประมวลผลทีละเฟรมและอัปเดตผลลัพธ์ทันที — กดสวิตช์อีกครั้งเพื่อหยุด"
    )

    # ขอหยุดกล้องจากรอบก่อนหน้า (เช่น เปิดกล้องไม่ได้ / อ่านเฟรมไม่ได้ / กดกลับหน้าแรก)
    if st.session_state.pop("force_stop_cam", False):
        st.session_state["cam_toggle"] = False
    if "cam_error" in st.session_state:
        st.error(st.session_state.pop("cam_error"))

    col_control, col_help = st.columns([1, 2], gap="large")

    with col_control:
        cam_index = st.number_input(
            "หมายเลขกล้อง", min_value=0, max_value=5, value=0, step=1
        )
        fps_target = st.slider("FPS เป้าหมาย", 1, 30, 10)
        downscale = st.toggle(
            "ลดขนาดภาพเพื่อความเร็ว", value=True,
            help="ย่อภาพให้กว้างไม่เกิน 640 px ก่อนประมวลผล",
        )
        running = st.toggle("▶ เริ่ม / หยุด Webcam", key="cam_toggle")

    with col_help:
        st.info(
            """
            **วิธีใช้งาน**
            1. เลือกหมายเลขกล้อง (ปกติคือ 0)
            2. เปิดสวิตช์ **เริ่ม / หยุด Webcam**
            3. ยกตัวเลขลายมือขึ้นไว้ที่ **กรอบสีขาวกลางจอ**
            4. ผลลัพธ์และกราฟความน่าจะเป็นจะอัปเดตทุกเฟรม
            5. ปิดสวิตช์เพื่อหยุดและคืนทรัพยากรกล้อง
            """
        )

    # --- ROI โฟกัสกลางจอ ---
    roi_frac = 0.40

    frame_slot = st.empty()
    digits_slot = st.empty()
    chart_slot = st.empty()
    status_slot = st.empty()

    # ── หยุดกล้อง ──
    if not running:
        release_camera()
        frame_slot.info("👆 เปิดสวิตช์ **เริ่ม / หยุด Webcam** เพื่อเริ่มต้น")
        return

    # ── เปิด/สลับกล้อง ──
    camera = st.session_state.get("cam")
    if camera is not None and st.session_state.get("cam_index") != int(cam_index):
        release_camera()
        camera = None

    if camera is None:
        camera = cv2.VideoCapture(int(cam_index))
        if not camera.isOpened():
            camera.release()
            st.session_state.cam = None
            st.session_state["force_stop_cam"] = True
            st.session_state["cam_error"] = (
                f"❌ เปิดกล้องหมายเลข {int(cam_index)} ไม่ได้ — "
                "ลองเปลี่ยนหมายเลขกล้องหรือตรวจสอบว่ากล้องไม่ถูกโปรแกรมอื่นใช้อยู่"
            )
            st.rerun()
        st.session_state.cam = camera
        st.session_state.cam_index = int(cam_index)

    # ── อ่านและประมวลผล 1 เฟรม ──
    started = time.time()
    ok, frame = camera.read()
    if not ok or frame is None:
        release_camera()
        st.session_state["force_stop_cam"] = True
        st.session_state["cam_error"] = "❌ อ่านภาพจากกล้องไม่ได้ — หยุดการทำงานแล้ว"
        st.rerun()

    if downscale and frame.shape[1] > 640:
        ratio = 640.0 / frame.shape[1]
        frame = cv2.resize(
            frame, (640, int(round(frame.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )

    # ตัด ROI ก่อนประมวลผล (ลด noise จากพื้นหลัง/เสื้อ)
    h, w = frame.shape[:2]
    roi_w = max(1, int(round(w * roi_frac)))
    roi_h = max(1, int(round(h * roi_frac)))
    r_x = int((w - roi_w) / 2)
    r_y = int((h - roi_h) / 2)
    r_x2 = r_x + roi_w
    r_y2 = r_y + roi_h
    roi_frame = frame[r_y:r_y2, r_x:r_x2]

    # ประมวลผลที่ ROI
    result = _run_with_settings(roi_frame, recognizer)

    # วาดกรอบ ROI บนภาพต้นฉบับ
    annotated = frame.copy()
    cv2.rectangle(annotated, (r_x, r_y), (r_x2, r_y2), (255, 255, 255), 2)
    cv2.putText(annotated, "Place number inside white box", (r_x, r_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, "Press toggle to start/stop", (r_x, r_y + roi_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame_slot.image(
        bgr_to_pil(annotated),
        caption=None,
        width="stretch",
    )

    if result.digits:
        tiles = "".join(
            f"""<div style="display:inline-block;margin:0 6px 6px 0;padding:10px 18px;
                background:linear-gradient(135deg,#4f46e5,#0ea5e9);border-radius:14px;
                color:#fff;font-size:1.9rem;font-weight:800;min-width:64px;
                text-align:center;">{d.digit}</div>"""
            for d in result.digits
        )
        digits_slot.markdown(
            f"""
            <div style="text-align:center;margin:10px 0;">
              {tiles}
              <p style="color:#475569;margin:4px 0 0;">
                ผลลัพธ์รวม:
                <strong style="font-size:1.35rem;color:#4f46e5;">{result.full_number}</strong>
              </p>
            </div>""",
            unsafe_allow_html=True,
        )

        focus = min(result.digits, key=lambda d: d.confidence)
        focus_n = next(
            (i + 1 for i, d in enumerate(result.digits) if d is focus), 1
        )
        chart_slot.markdown(
            prob_chart_html(
                focus.probabilities,
                f"ความน่าจะเป็น — หลักที่ {focus_n} (ทำนาย {focus.digit}, "
                f"มั่นใจ {focus.confidence * 100:.1f}%)",
            ),
            unsafe_allow_html=True,
        )
    else:
        digits_slot.info("ยังไม่พบตัวเลขใน ROI — ย้ายตัวเลขให้อยู่ในกรอบสีขาว")
        chart_slot.empty()

    elapsed = time.time() - started
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    status_slot.caption(
        f"🟢 กำลังทำงาน · กล้อง {int(cam_index)} · ROI กลาง {roi_w}x{roi_h} · ประมวลผล {fps:.1f} FPS (เป้าหมาย {fps_target} FPS)"
    )

    # ── ควบคุม FPS แล้ววนเฟรมถัดไป ──
    delay = max(0.0, (1.0 / fps_target) - elapsed)
    if delay > 0:
        time.sleep(delay)
    st.rerun()


# ═══════════════════════════════════════════════════════
#  Router
# ═══════════════════════════════════════════════════════
if st.session_state.mode == MODE_IMAGE:
    render_image_mode()
elif st.session_state.mode == MODE_WEBCAM:
    render_webcam_mode()
else:
    render_menu()