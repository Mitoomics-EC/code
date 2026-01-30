import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
import warnings
import os

warnings.filterwarnings('ignore')

TARGET_CLASSES = ['鳞癌', '腺癌']

FOLDS = 5
RANDOM_STATE = 47  # 🟢 固定随机种子为 47
COST_0 = 1.5  # 腺癌权重
COST_1 = 1.0  # 鳞癌权重

file_path = r"D:/PSP input.xlsx"
output_dir = r"D:/PSP"

print(f"{'=' * 60}")
print("🔍 正在加载数据...")

try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print(f"✅ 成功读取文件: {file_path}")
    print(f"📊 数据形状: {data.shape}")
except Exception as e:
    print(f"❌ 文件读取失败: {e}")
    exit()

y_raw_col = data.iloc[:, 0]
y_clean = y_raw_col.astype(str).str.strip().values
unique_labels = np.unique(y_clean)

print(f"🧐 Excel 中发现的所有类别名称: {unique_labels}")
print(f"🎯 代码寻找的目标类别: {TARGET_CLASSES}")

mask = np.isin(y_clean, TARGET_CLASSES)
X = data.iloc[:, 1:].values[mask]
y = y_clean[mask]
original_index = data.index[mask]

if len(y) == 0:
    print(f"\n{'❌' * 20}")
    print("严重错误：筛选后没有剩下任何样本！")
    print("请修改代码顶部的 TARGET_CLASSES 以匹配 Excel 中的名称。")
    exit()

print(f"✅ 筛选成功！有效样本数: {len(y)}")
print(f"   类别分布: {pd.Series(y).value_counts().to_dict()}")


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

pos_key = next((k for k in label_mapping if '鳞' in k or 'Squamous' in k or 'ESCC' in k), None)
if pos_key and label_mapping[pos_key] == 0:
    y_encoded = 1 - y_encoded
    label_mapping = {k: 1 - v for k, v in label_mapping.items()}
    print(f"🔄 自动修正标签方向: {label_mapping}")
else:
    print(f"✅ 标签映射确认: {label_mapping}")

def clean_data(X):
    X = np.array(X, dtype=np.float64)
    X[np.isinf(X)] = np.nan
    if np.any(np.isnan(X)):
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    return X


X = clean_data(X)


def get_weights(y_train, c0, c1):
    c0_n = np.sum(y_train == 0)
    c1_n = np.sum(y_train == 1)
    n = len(y_train)
    w0 = (n / (2 * c0_n)) * c0 if c0_n > 0 else 1.0
    w1 = (n / (2 * c1_n)) * c1 if c1_n > 0 else 1.0
    return {0: w0, 1: w1}


os.makedirs(output_dir, exist_ok=True)
print(f"\n🚀 开始 Logit 训练 (Seed={RANDOM_STATE}, Cost0={COST_0}, Cost1={COST_1})...")

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = []
all_probs = np.zeros(len(y))

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_encoded)):
    Xt, Xv = X[tr_idx], X[te_idx]
    yt, yv = y_encoded[tr_idx], y_encoded[te_idx]

    sc = StandardScaler()
    Xt = sc.fit_transform(Xt)
    Xv = sc.transform(Xv)


    cw = get_weights(yt, COST_0, COST_1)
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight=cw,
        solver='liblinear',
        C=1.0,
        max_iter=2000
    )
    model.fit(Xt, yt)

    y_p = model.predict_proba(Xv)[:, 1]
    all_probs[te_idx] = y_p
    auc = roc_auc_score(yv, y_p)
    fpr, tpr, ths = roc_curve(yv, y_p)

    best_th = ths[np.argmax(tpr - fpr)] if len(ths) > 0 else 0.5
    y_pred_bin = (y_p >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, y_pred_bin).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"   -> Fold {fold + 1}: AUC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    fold_metrics.append({
        'Fold': fold + 1,
        'AUC': auc,
        'Sens': sens,
        'Spec': spec,
        'Threshold': best_th
    })

mean_auc = np.mean([x['AUC'] for x in fold_metrics])
mean_sens = np.mean([x['Sens'] for x in fold_metrics])
mean_spec = np.mean([x['Spec'] for x in fold_metrics])

print(f"\n✅ 最终结果 (Seed {RANDOM_STATE} - 5折平均):")
print(f"   AUC : {mean_auc:.4f}")
print(f"   Sens: {mean_sens:.4f}")
print(f"   Spec: {mean_spec:.4f}")

summary_data = [{
    'Model': 'Logit_CostSensitive',
    'Seed': RANDOM_STATE,
    'AUC_Mean': mean_auc,
    'Sens_Mean': mean_sens,
    'Spec_Mean': mean_spec,
    'Cost_0': COST_0,
    'Cost_1': COST_1
}]

df_res = pd.DataFrame(index=original_index)
df_res['True_Label'] = y
df_res['True_Name'] = np.where(y_encoded == 1, '鳞癌', '腺癌')
df_res['Logit_Prob'] = all_probs

path = f"{output_dir}/Logit_Seed{RANDOM_STATE}_Result.xlsx"
with pd.ExcelWriter(path) as writer:
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    pd.DataFrame(fold_metrics).to_excel(writer, sheet_name='Fold_Details', index=False)
    df_res.to_excel(writer, sheet_name='Predictions')

print(f"\n🎉 成功！结果已保存: {path}")