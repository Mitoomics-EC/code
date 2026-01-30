# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression  # PLS-DA核心库
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
import warnings
from scipy import stats
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

file_path = r"train.xlsx"
try:
    data = pd.read_excel(file_path, engine='openpyxl')
    print("数据加载成功！")
    print(f"数据形状: {data.shape}")

    sample_ids = data.iloc[:, 0].values
    print(f"样本编号列名称: {data.columns[0]}")
    print(f"前5个样本编号: {sample_ids[:5]}")
    data = data.iloc[:, 1:]
    print(f"删除样本编号后数据形状: {data.shape}")

except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

print("\n数据前5行（已删除样本编号列）:")
print(data.head())

print("\n数据基本信息:")
print(data.info())

print("\n检查缺失值:")
print(data.isnull().sum())

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\n原始标签映射:", label_mapping)

target_mapping = {'EC': 1, 'controls': 0}
current_ec_code = label_mapping.get('EC', None)
current_controls_code = label_mapping.get('controls', None)

if current_ec_code == 0 and current_controls_code == 1:
    print("反转标签编码，使EC=1（阳性），controls=0（阴性）")
    y = 1 - y
    label_mapping = target_mapping
elif current_ec_code == 1 and current_controls_code == 0:
    print("标签编码正确，EC=1（阳性），controls=0（阴性）")
else:
    print(f"警告：检测到未预期的标签类别 {label_encoder.classes_}")
    print("强制设置标签映射为 EC=1，controls=0")
    y = np.where(data.iloc[:, 0].values == 'EC', 1, 0)
    label_mapping = target_mapping

print("最终标签映射:", label_mapping)
print(f"EC样本数: {sum(y == 1)}, controls样本数: {sum(y == 0)}")


def clean_data(X):
    """处理NaN、Inf和非数值型数据"""
    X = np.array(X, dtype=np.float64)
    X[np.isinf(X)] = np.nan
    if np.any(np.isnan(X)):
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    return X


X = clean_data(X)

class PLSDAClassifier:
    """PLS-DA分类器封装，支持scikit-learn的clone操作"""

    def __init__(self, n_components=5, random_state=24):
        self.n_components = n_components
        self.random_state = random_state
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = [0, 1]

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.pls = PLSRegression(n_components=self.n_components)
        return self

    def fit(self, X, y):

        y_dummy = np.zeros((len(y), 2))
        y_dummy[np.arange(len(y)), y] = 1
        self.pls.fit(X, y_dummy)
        return self

    def predict_proba(self, X):
        """预测概率：返回[P(0), P(1)]"""
        y_pred = self.pls.predict(X)
        y_proba = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return y_proba

    def predict(self, X):
        """预测标签：取概率最大的类别"""
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)

models = {
    'PLS-DA': PLSDAClassifier(n_components=5, random_state=24)
}

def evaluate_model_with_cv(model, X, y, model_name, sample_ids):
    """使用十折交叉验证评估模型性能"""
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=24)

    auc_scores = []
    sensitivity_scores = []
    specificity_scores = []
    all_sample_probas = np.zeros(len(y))
    all_sample_folds = np.zeros(len(y), dtype=int) - 1

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = clean_data(X_train)
        X_test = clean_data(X_test)

        if model_name in ['PLS-DA']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        try:
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            auc = roc_auc_score(y_test, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            auc_scores.append(auc)
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
            all_sample_probas[test_idx] = y_pred_proba
            all_sample_folds[test_idx] = fold

            print(f"{model_name} 第{fold + 1}折完成: AUC={auc:.4f}, 敏感性={sensitivity:.4f}, 特异性={specificity:.4f}")

        except Exception as e:
            print(f"在 {model_name} 的第{fold + 1}折中遇到错误: {e}")
            continue

    def calculate_ci(data):
        if len(data) > 1:
            return stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data))
        else:
            return (np.mean(data), np.mean(data))

    return {
        'AUC_mean': np.mean(auc_scores),
        'AUC_ci': calculate_ci(auc_scores),
        'Sensitivity_mean': np.mean(sensitivity_scores),
        'Sensitivity_ci': calculate_ci(sensitivity_scores),
        'Specificity_mean': np.mean(specificity_scores),
        'Specificity_ci': calculate_ci(specificity_scores),
        'AUC_scores': auc_scores,
        'Sensitivity_scores': sensitivity_scores,
        'Specificity_scores': specificity_scores,
        'sample_probas': all_sample_probas,
        'sample_folds': all_sample_folds,
        'sample_ids': sample_ids
    }


results = {}
for name, model in models.items():
    print(f"\n正在评估 {name} 模型 (10折交叉验证)...")
    results[name] = evaluate_model_with_cv(model, X, y, name, sample_ids)

model_names = list(results.keys())

auc_means = [results[name]['AUC_mean'] for name in model_names]
auc_cis = [results[name]['AUC_ci'] for name in model_names]
sens_means = [results[name]['Sensitivity_mean'] for name in model_names]
sens_cis = [results[name]['Sensitivity_ci'] for name in model_names]
spec_means = [results[name]['Specificity_mean'] for name in model_names]
spec_cis = [results[name]['Specificity_ci'] for name in model_names]


auc_err = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(auc_means, auc_cis)]
sens_err = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(sens_means, sens_cis)]
spec_err = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(spec_means, spec_cis)]

output_dir = r"ED model results"
os.makedirs(output_dir, exist_ok=True)

sample_scores_df = pd.DataFrame({
    '样本编号': results[model_names[0]]['sample_ids'],
    '真实标签': y,
    '真实标签名称': np.where(y == 1, 'EC', 'controls')
})

for name in model_names:
    sample_scores_df[f'{name}_EC预测概率'] = results[name]['sample_probas']

sample_scores_df.to_excel(f"{output_dir}/PLS-DA样本预测得分表_含样本编号.xlsx", index=False)

print("\n===== PLS-DA模型性能摘要（EC vs controls，10折交叉验证） =====")
for name in model_names:
    auc_mean = results[name]['AUC_mean']
    auc_ci = results[name]['AUC_ci']

    print(f"{name}: "
          f"AUC={auc_mean:.4f} (95%CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})")

print(f"\n所有结果已保存至: {output_dir}")
