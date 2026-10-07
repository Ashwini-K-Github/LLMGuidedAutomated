import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def evaluate_classification_model_debug(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    unique_classes = np.unique(y_test)
    is_binary = len(unique_classes) == 2
    
    if is_binary:
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = 0.5
        if y_probs is not None and y_probs.shape[1] == 2:
            try:
                roc_auc = float(roc_auc_score(y_test, y_probs[:, 1]))
            except Exception as e:
                print(f"ROC-AUC calculation error: {e}")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = 0, 0, 0, 0
        cm_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    else:
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        
        roc_auc = 0.5
        print("Initial roc_auc:", roc_auc)
        if y_probs is not None:
            try:
                roc_auc = float(roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro"))
                print("After first try:", roc_auc)
            except Exception as e:
                print("First try failed:", e)
                try:
                    roc_auc = float(roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro"))
                    print("After second try:", roc_auc)
                except Exception as e2:
                    print("Second try failed:", e2)
                    pass
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {"matrix": cm.tolist(), "labels": [str(c) for c in unique_classes]}
        
    res = {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1_score": round(f1, 3),
        "roc_auc": round(roc_auc, 3),
        "confusion_matrix": cm_dict,
        "is_binary": is_binary
    }
    print("Before nan/inf check:", res)
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        val = res[k]
        if np.isnan(val) or np.isinf(val):
            res[k] = 0.0
    return res

class MockModel:
    def predict(self, X):
        return np.array([1, 1])
    def predict_proba(self, X):
        return np.array([[0.1, 0.8, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05]])

X_test = np.array([[1, 2], [3, 4]])
y_test = np.array([1, 1])

res = evaluate_classification_model_debug(MockModel(), X_test, y_test)
print("Final Result:", res)
