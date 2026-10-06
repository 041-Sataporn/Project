# วางเป็นเซลล์สุดท้ายใน Notebook (ต่อจากเซลล์ที่ 28 ที่เทรน nn_model เสร็จแล้ว) แล้วรัน
# จะได้ไฟล์ scaler.joblib และ fraud_model.joblib ให้ดาวน์โหลดจาก Colab
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

y_prob = y_pred_nn_prob.reshape(-1)
y_hat = (y_prob >= 0.5).astype(int)

layers = []
for layer in nn_model.layers:
    w = layer.get_weights()
    if len(w) == 2:  # เฉพาะ Dense (Dropout ไม่มีน้ำหนัก และไม่ทำงานตอนทำนาย)
        layers.append({"W": w[0], "b": w[1], "activation": layer.activation.__name__})

bundle = {
    "layers": layers,
    "threshold": 0.5,
    "features": list(X_train.columns),
    "metrics": {
        "accuracy": accuracy_score(y_test, y_hat),
        "precision": precision_score(y_test, y_hat, zero_division=0),
        "recall": recall_score(y_test, y_hat),
        "f1": f1_score(y_test, y_hat, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    },
}

joblib.dump(scaler, "scaler.joblib")        # scaler ตัวที่ใช้กับ nn_model (เซลล์ 28)
joblib.dump(bundle, "fraud_model.joblib")
print(bundle["metrics"])

from google.colab import files
files.download("scaler.joblib")
files.download("fraud_model.joblib")
