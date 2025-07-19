import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# === Load the data ===
X = np.load("processed_data/X_EC_enhanced.npy", allow_pickle=True)
y = np.load("processed_data/y_EC.npy", allow_pickle=True)
subject_ids = np.load("processed_data/subject_ids_EC.npy", allow_pickle=True)

# === Ensure correct shape ===
X = X.reshape((X.shape[0], -1))
y = y.reshape(-1)
subject_ids = subject_ids.reshape(-1)

print(f"✅ Loaded: X = {X.shape}, y = [0: {(y==0).sum()} | 1: {(y==1).sum()}]")

# === Cross-validation setup ===
sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_f1s = []
fold_accs = []
all_preds = []
all_targets = []
all_subjects = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=subject_ids), 1):
    print(f"\n🔁 Fold {fold}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_subjects = subject_ids[test_idx]

    # === Apply SMOTE ===
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"📊 After SMOTE: {Counter(y_train_res)}")

    # === Train model ===
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    # === Evaluate ===
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Fold {fold} - F1: {f1:.4f}, Acc: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    fold_f1s.append(f1)
    fold_accs.append(acc)

    all_preds.extend(y_pred)
    all_targets.extend(y_test)
    all_subjects.extend(test_subjects)

# === Subject-level evaluation ===
from collections import defaultdict

subject_preds = defaultdict(list)
subject_targets = {}

for pred, true, subj in zip(all_preds, all_targets, all_subjects):
    subject_preds[subj].append(pred)
    subject_targets[subj] = true  # assumes true label same per subject

final_preds = []
final_truths = []

for subj in subject_preds:
    majority_vote = int(np.round(np.mean(subject_preds[subj])))
    final_preds.append(majority_vote)
    final_truths.append(subject_targets[subj])

print("\n📊 SUBJECT-LEVEL EVALUATION")
print(f"✅ Accuracy: {accuracy_score(final_truths, final_preds):.4f}")
print(f"✅ F1 Score: {f1_score(final_truths, final_preds):.4f}")
print("✅ Classification Report:\n", classification_report(final_truths, final_preds))
