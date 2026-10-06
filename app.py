import glob
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- ไฟล์ที่แอปใช้ (วางไว้โฟลเดอร์เดียวกับ app.py) ----------
SCALER_PATH = "scaler.joblib"        # StandardScaler ที่ใช้ตอนเทรน
MODEL_PATH = "fraud_model.joblib"    # น้ำหนักโมเดล + ค่าความแม่นยำ (สร้างจาก export_model_cell.py)

DEFAULT_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEVELOPERS = [
    "039 ศักดิโชติ แตงโสภา",
    "041 สถาพร ขวาธิจักร",
    "044 อติชาต พันธุ์ขะวงษ์",
]

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 820px; padding-top: 2.5rem;}
    h1 {font-weight: 600; letter-spacing: -0.5px;}
    .subtitle {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1.5rem;}
    .card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 0.6rem;
           text-align: center; background: #fafafa;}
    .card .label {color: #6b7280; font-size: 0.75rem; text-transform: uppercase;
                  letter-spacing: 0.06em;}
    .card .value {font-size: 1.5rem; font-weight: 600; color: #111827;}
    .result {border-radius: 12px; padding: 1.2rem 1.4rem; margin-top: 1rem;
             font-size: 1.05rem; border: 1px solid;}
    .ok {background: #f0fdf4; border-color: #bbf7d0; color: #166534;}
    .bad {background: #fef2f2; border-color: #fecaca; color: #991b1b;}
    .footer {text-align: center; color: #6b7280; font-size: 0.9rem;
             border-top: 1px solid #e5e7eb; padding-top: 1rem; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- โหลดโมเดล ----------
def _find(path: str, pattern: str):
    """หาไฟล์ตามชื่อ ถ้าไม่เจอให้ลองหาตาม pattern ในโฟลเดอร์เดียวกับ app.py"""
    base = os.path.dirname(os.path.abspath(__file__))
    for p in (path, os.path.join(base, path)):
        if os.path.exists(p):
            return p
    found = sorted(glob.glob(os.path.join(base, pattern)))
    return found[0] if found else None


class FraudModel:
    """Neural network (MLP) ที่รันด้วย NumPy ล้วน ไม่ต้องติดตั้ง TensorFlow"""

    def __init__(self, scaler, bundle: dict):
        self.scaler = scaler
        self.layers = bundle["layers"]
        self.threshold = float(bundle.get("threshold", 0.5))
        self.metrics = bundle.get("metrics", {}) or {}
        self.features = list(
            getattr(scaler, "feature_names_in_", bundle.get("features", DEFAULT_FEATURES))
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(df[self.features].astype(float).values)
        for layer in self.layers:
            x = x @ np.asarray(layer["W"]) + np.asarray(layer["b"])
            act = layer.get("activation", "linear")
            if act == "relu":
                x = np.maximum(x, 0.0)
            elif act == "sigmoid":
                x = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return x.reshape(-1)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    s_path = _find(SCALER_PATH, "*scaler*.joblib")
    m_path = _find(MODEL_PATH, "*model*.joblib")
    if s_path is None:
        return None, "ไม่พบไฟล์ scaler.joblib"
    if m_path is None:
        return None, "ไม่พบไฟล์ fraud_model.joblib (น้ำหนักโมเดล)"
    return FraudModel(joblib.load(s_path), joblib.load(m_path)), None


model, error = load_model()

# ---------- ส่วนหัว ----------
st.title("💳 ระบบตรวจจับการทุจริตบัตรเครดิต")
st.markdown(
    '<div class="subtitle">Credit Card Fraud Detection · Deep Learning (Neural Network)</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.error(error)
    st.stop()


def show_single(prob: float):
    is_fraud = prob >= model.threshold
    css = "bad" if is_fraud else "ok"
    text = "⚠️ ตรวจพบความเสี่ยงทุจริต" if is_fraud else "✅ ธุรกรรมปกติ"
    st.markdown(
        f'<div class="result {css}"><b>{text}</b><br>'
        f"ความน่าจะเป็นที่เป็น Fraud: {prob:.2%} (เกณฑ์ตัดสิน {model.threshold:.2f})</div>",
        unsafe_allow_html=True,
    )


# ---------- ตรวจสอบธุรกรรม ----------
tab_file, tab_manual = st.tabs(["อัปโหลดไฟล์ CSV", "กรอกข้อมูลทีละรายการ"])

with tab_file:
    st.caption("ไฟล์ต้องมีคอลัมน์ Time, V1–V28, Amount (คอลัมน์ Class ถ้ามีจะถูกข้าม)")
    up = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in model.features if c not in data.columns]
        if missing:
            st.error(f"ไม่พบคอลัมน์: {', '.join(missing)}")
        else:
            probs = model.predict_proba(data)
            out = data.copy()
            out["fraud_probability"] = probs
            out["prediction"] = np.where(probs >= model.threshold, "Fraud", "Normal")

            c1, c2 = st.columns(2)
            c1.metric("จำนวนธุรกรรมทั้งหมด", f"{len(out):,}")
            c2.metric("ตรวจพบ Fraud", f"{int((out['prediction'] == 'Fraud').sum()):,}")

            st.dataframe(
                out.sort_values("fraud_probability", ascending=False).head(200),
                use_container_width=True,
            )
            st.download_button(
                "ดาวน์โหลดผลลัพธ์ (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv",
                "text/csv",
            )

with tab_manual:
    c1, c2 = st.columns(2)
    t_val = c1.number_input("Time (วินาที)", value=0.0)
    a_val = c2.number_input("Amount (จำนวนเงิน)", value=0.0, min_value=0.0)
    with st.expander("ค่า V1–V28 (ผลจาก PCA)"):
        cols = st.columns(4)
        v_vals = [cols[i % 4].number_input(f"V{i + 1}", value=0.0, key=f"v{i + 1}") for i in range(28)]
    if st.button("ตรวจสอบ", type="primary"):
        row = {"Time": t_val, "Amount": a_val, **{f"V{i + 1}": v for i, v in enumerate(v_vals)}}
        show_single(float(model.predict_proba(pd.DataFrame([row]))[0]))

# ---------- ความแม่นยำ ----------
st.markdown("### ความแม่นยำของระบบ")
m = model.metrics
labels = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-score"),
    ("auc", "ROC-AUC"),
]
items = [(lab, m[k]) for k, lab in labels if k in m]
if items:
    cols = st.columns(len(items))
    for col, (lab, val) in zip(cols, items):
        col.markdown(
            f'<div class="card"><div class="label">{lab}</div>'
            f'<div class="value">{val:.2%}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"ประเมินจากชุดข้อมูลทดสอบ 20% (Precision / Recall / F1 คิดจากคลาส Fraud) "
        f"ที่เกณฑ์ตัดสิน {model.threshold:.2f}"
    )
else:
    st.info("ไม่พบค่าความแม่นยำในไฟล์โมเดล กรุณาสร้างไฟล์ด้วย export_model_cell.py")

# ---------- ส่วนท้าย ----------
st.markdown(
    '<div class="footer"><b>ผู้พัฒนา</b><br>' + "<br>".join(DEVELOPERS) + "</div>",
    unsafe_allow_html=True,
)
