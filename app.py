import glob
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "039_041_044_ใบงานที่6 (4).joblib"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEVELOPERS = [
    "039 ศักดิโชติ แตงโสภา",
    "041 สถาพร ขวาธิจักร",
    "044 อติชาต พันธุ์ขะวงษ์",
]

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered",
)

# ---------- Minimal style ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 820px; padding-top: 2.5rem;}
    h1 {font-weight: 600; letter-spacing: -0.5px;}
    .subtitle {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1.5rem;}
    .card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.2rem;
           text-align: center; background: #fafafa;}
    .card .label {color: #6b7280; font-size: 0.8rem; text-transform: uppercase;
                  letter-spacing: 0.06em;}
    .card .value {font-size: 1.6rem; font-weight: 600; color: #111827;}
    .result {border-radius: 12px; padding: 1.2rem 1.4rem; margin-top: 1rem;
             font-size: 1.1rem; border: 1px solid;}
    .ok {background: #f0fdf4; border-color: #bbf7d0; color: #166534;}
    .bad {background: #fef2f2; border-color: #fecaca; color: #991b1b;}
    .footer {text-align: center; color: #6b7280; font-size: 0.9rem;
             border-top: 1px solid #e5e7eb; padding-top: 1rem; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model ----------
class ModelWrapper:
    """ห่อโมเดลให้เรียกใช้เหมือนกันทุกรูปแบบ: predict_proba(df) -> ความน่าจะเป็น Fraud"""

    def __init__(self, bundle):
        self.scaler = None
        self.threshold = 0.3
        self.metrics = {}
        self.features = FEATURES
        self.keras_model = None
        self.sk_model = None

        if isinstance(bundle, dict):
            self.scaler = bundle.get("scaler")
            self.threshold = float(bundle.get("threshold", 0.3))
            self.metrics = bundle.get("metrics", {}) or {}
            self.features = bundle.get("features", FEATURES)

            if "model_json" in bundle and "weights" in bundle:
                import tensorflow as tf

                self.keras_model = tf.keras.models.model_from_json(bundle["model_json"])
                self.keras_model.set_weights(bundle["weights"])
            elif "model" in bundle:
                self._set_model(bundle["model"])
            else:
                raise ValueError("ไม่พบโมเดลใน joblib (ต้องมี model หรือ model_json+weights)")
        else:
            self._set_model(bundle)

    def _set_model(self, model):
        if hasattr(model, "predict_proba"):
            self.sk_model = model
        else:
            self.keras_model = model

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].astype(float).values
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.keras_model is not None:
            return np.asarray(self.keras_model.predict(x, verbose=0)).reshape(-1)
        return self.sk_model.predict_proba(x)[:, 1]


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    path = MODEL_PATH
    if not os.path.exists(path):
        # GitHub/ระบบอาจเปลี่ยนชื่อไฟล์ -> ใช้ไฟล์ .joblib ตัวแรกที่เจอในโฟลเดอร์
        found = sorted(glob.glob(os.path.join(os.path.dirname(__file__) or ".", "*.joblib")))
        if not found:
            return None
        path = found[0]
    return ModelWrapper(joblib.load(path))


model = load_model()

# ---------- Header ----------
st.title("💳 ระบบตรวจจับการทุจริตบัตรเครดิต")
st.markdown(
    '<div class="subtitle">Credit Card Fraud Detection · Deep Learning</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.error(f"ไม่พบไฟล์โมเดล `{MODEL_PATH}` กรุณาวางไฟล์ไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

# ---------- Prediction ----------
tab_file, tab_manual = st.tabs(["อัปโหลดไฟล์ CSV", "กรอกข้อมูลทีละรายการ"])


def show_single(prob: float):
    is_fraud = prob >= model.threshold
    css = "bad" if is_fraud else "ok"
    text = "⚠️ ตรวจพบความเสี่ยงทุจริต" if is_fraud else "✅ ธุรกรรมปกติ"
    st.markdown(
        f'<div class="result {css}"><b>{text}</b><br>'
        f"ความน่าจะเป็นที่เป็น Fraud: {prob:.2%} "
        f"(เกณฑ์ตัดสิน {model.threshold:.2f})</div>",
        unsafe_allow_html=True,
    )


with tab_file:
    st.caption(
        "ไฟล์ต้องมีคอลัมน์ Time, V1–V28, Amount (คอลัมน์ Class ถ้ามีจะถูกข้าม)"
    )
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

            n_fraud = int((out["prediction"] == "Fraud").sum())
            c1, c2 = st.columns(2)
            c1.metric("จำนวนธุรกรรมทั้งหมด", f"{len(out):,}")
            c2.metric("ตรวจพบ Fraud", f"{n_fraud:,}")

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
    st.caption("กรอกค่า Amount และ Time ส่วน V1–V28 ใส่เป็น 0 หรือวางค่าทั้งแถว")
    row_text = st.text_area(
        "วางข้อมูล 1 แถว (30 ค่า คั่นด้วยจุลภาค: Time, V1..V28, Amount)",
        placeholder="0,-1.35,-0.07,2.53,...,149.62",
        height=90,
    )
    if st.button("ตรวจสอบ", type="primary"):
        try:
            vals = [float(v) for v in row_text.replace("\n", ",").split(",") if v.strip()]
            if len(vals) != len(model.features):
                st.error(f"ต้องมี {len(model.features)} ค่า แต่ได้รับ {len(vals)} ค่า")
            else:
                df1 = pd.DataFrame([vals], columns=model.features)
                show_single(float(model.predict_proba(df1)[0]))
        except ValueError:
            st.error("รูปแบบข้อมูลไม่ถูกต้อง กรุณาใส่ตัวเลขคั่นด้วยจุลภาค")

# ---------- Model performance ----------
st.markdown("### ความแม่นยำของระบบ")
m = model.metrics
if m:
    labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision (Fraud)"),
        ("recall", "Recall (Fraud)"),
        ("f1", "F1-score (Fraud)"),
        ("auc", "ROC-AUC"),
    ]
    items = [(lab, m[k]) for k, lab in labels if k in m]
    cols = st.columns(len(items))
    for col, (lab, val) in zip(cols, items):
        col.markdown(
            f'<div class="card"><div class="label">{lab}</div>'
            f'<div class="value">{val:.2%}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"ประเมินจากชุดข้อมูลทดสอบ (20% ของข้อมูล) ที่เกณฑ์ตัดสิน {model.threshold:.2f}"
    )
else:
    st.info("ยังไม่มีค่าความแม่นยำในไฟล์โมเดล — ดูวิธีบันทึกในคู่มือ")

# ---------- Footer ----------
st.markdown(
    '<div class="footer"><b>ผู้พัฒนา</b><br>'
    + "<br>".join(DEVELOPERS)
    + "</div>",
    unsafe_allow_html=True,
)
