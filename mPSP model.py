import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import statsmodels.api as sm

import os
import warnings

warnings.filterwarnings("ignore")

# ================= ⚙️ 全局配置 =================
INPUT_XLSX = r"D:/ESCC and EAC.xlsx"
OUTPUT_DIR = r"D:/Result"

# 随机种子设置
FIXED_SEED = 77

# 定义：C_FP = 误报代价 (把腺癌判成鳞癌), C_FN = 漏报代价 (把鳞癌判成腺癌)
COST_FP = 1.5
COST_FN = 1.0

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def to_binary(y_raw):
    """标签转换"""
    y_series = pd.Series(y_raw).astype(str).str.strip().str.upper()
    is_escc = y_series.str.contains("SQUAMOUS") | \
              y_series.str.contains("ESCC") | \
              y_series.str.contains("鳞癌")
    return np.where(is_escc, 1, 0)


def get_cost_sensitive_weight(y, c_fp, c_fn):
    """
    [核心修改] 计算代价敏感权重
    公式：Final_Weight = (Count_Neg / Count_Pos) * (Cost_FN / Cost_FP)

    解释：
    1. (Count_Neg / Count_Pos): 基础平衡项，消除数据不平衡的影响。
    2. (Cost_FN / Cost_FP):     代价惩罚项，人为增加对高代价错误的惩罚。
    """
    count_0 = np.sum(y == 0)
    count_1 = np.sum(y == 1)
    if count_1 == 0: return 1.0

    balance_ratio = count_0 / count_1
    cost_ratio = c_fn / c_fp

    return balance_ratio * cost_ratio


class BinaryFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.encoders = {}
        self.feature_names = feature_names

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names) if isinstance(X, np.ndarray) else X
        for col in X_df.columns:
            le = LabelEncoder()
            le.fit(X_df[col].astype(str).fillna("Missing"))
            self.encoders[col] = le
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names) if isinstance(X, np.ndarray) else X
        encoded_cols = [self.encoders[col].transform(X_df[col].astype(str).fillna("Missing")).reshape(-1, 1) for col in
                        X_df.columns]
        return np.hstack(encoded_cols)


def create_model(algorithm, seed, pos_weight):
    """
    [核心修改] 将所有模型的 class_weight 设置为基于代价的显式权重
    """
    class_weights_dict = {0: 1.0, 1: pos_weight}

    if algorithm == 'lr':

        return LogisticRegression(class_weight=class_weights_dict, solver='liblinear', C=0.4, random_state=seed,
                                  max_iter=2000)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")


def run_experiment(algo_name):
    print(f"\n{'=' * 10} 正在运行: {algo_name.upper()} (代价敏感模式 Cost_FN={COST_FN}) {'=' * 10}")

    df = pd.read_excel(INPUT_XLSX)
    df.columns = [str(c).strip() for c in df.columns]

    sample_id, label_col = df.columns[0], df.columns[1]

    mtDNA_col = next((c for c in df.columns if "MTDNA" in c.upper() and "SCORE" in c.upper()), None)
    if not mtDNA_col:
        mtDNA_col = next((c for c in df.columns if "MTDNA" in c.upper()), "mtDNA_score")

    clinical_cols = [c for c in df.columns if c not in [sample_id, label_col, mtDNA_col]]
    numeric_cols = [c for c in clinical_cols if pd.api.types.is_numeric_dtype(df[c])]
    binary_cols = [c for c in clinical_cols if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == 2]
    multi_cols = [c for c in clinical_cols if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]

    y = to_binary(df[label_col])

    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
    print(f"标签: 鳞癌(1)={n_pos} | 腺癌(0)={n_neg}")
    if n_pos == 0 or n_neg == 0: return

    # 灰色地带定义
    lower, upper = np.percentile(df[mtDNA_col][y == 1], 1.5), np.percentile(df[mtDNA_col][y == 0], 97.5)
    real_lower, real_upper = min(lower, upper), max(lower, upper)
    grey_mask = (df[mtDNA_col] >= real_lower) & (df[mtDNA_col] <= real_upper)

    X_rescue, y_rescue = df.loc[grey_mask].copy(), y[grey_mask]

    pos_weight = get_cost_sensitive_weight(y_rescue, COST_FP, COST_FN)

    print(f"灰色地带: [{real_lower:.4f}, {real_upper:.4f}] | 样本数: {len(y_rescue)}")
    print(f"⚖️ 代价敏感权重 (Pos Weight): {pos_weight:.4f} (基准平衡 x 代价因子 {COST_FN / COST_FP})")

    num_pipeline = Pipeline([('imp', SimpleImputer(strategy='median')), ('std', StandardScaler())])
    bin_pipeline = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                             ('enc', BinaryFeatureTransformer(feature_names=binary_cols))])
    mul_pipeline = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                             ('one', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, [mtDNA_col] + numeric_cols),
        ('bin', bin_pipeline, binary_cols),
        ('mul', mul_pipeline, multi_cols)
    ])

    X_res_proc = preprocessor.fit_transform(X_rescue)

    final_feat_names = [mtDNA_col] + numeric_cols + binary_cols
    if len(multi_cols) > 0:
        try:
            final_feat_names += preprocessor.named_transformers_['mul'].named_steps['one'].get_feature_names_out(
                multi_cols).tolist()
        except:
            final_feat_names += [f"Multi_{i}" for i in range(X_res_proc.shape[1] - len(final_feat_names))]

    seed_results = []
    print("🚀 开始模型拟合")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FIXED_SEED)
    seed = FIXED_SEED
    aucs = []
    try:
        for t, v in skf.split(X_res_proc, y_rescue):
            m = create_model(algo_name, seed, pos_weight)
            m.fit(X_res_proc[t], y_rescue[t])
            aucs.append(roc_auc_score(y_rescue[v], m.predict_proba(X_res_proc[v])[:, 1]))
        seed_results.append({"Seed": seed, "Mean_AUC": np.mean(aucs)})
    except Exception:
        print("Modeling eval failed!")

    if not seed_results: return

    skf_fix = StratifiedKFold(n_splits=5, shuffle=True, random_state=FIXED_SEED)
    grey_probs = np.zeros(len(y_rescue))

    for t, v in skf_fix.split(X_res_proc, y_rescue):
        m_fix = create_model(algo_name, FIXED_SEED, pos_weight)
        m_fix.fit(X_res_proc[t], y_rescue[t])
        grey_probs[v] = m_fix.predict_proba(X_res_proc[v])[:, 1]

    df["Integrated_Score"] = df[mtDNA_col]
    df.loc[grey_mask, "Integrated_Score"] = grey_probs

    output_path = os.path.join(OUTPUT_DIR, f"Final_Result_{algo_name}.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="Integrated_Data", index=False)
        pd.DataFrame(seed_results).to_excel(writer, sheet_name="Seed_History", index=False)

        auc_vals = [x['Mean_AUC'] for x in seed_results]
        pd.DataFrame({
            "Metric": ["Max_AUC", "Avg_AUC", "Std_Dev", "GreyZone_Lower", "GreyZone_Upper", "Cost_FN", "Cost_FP"],
            "Value": [np.max(auc_vals), np.mean(auc_vals), np.std(auc_vals), real_lower, real_upper, COST_FN, COST_FP]
        }).to_excel(writer, sheet_name="Summary", index=False)

        m_imp = create_model(algo_name, FIXED_SEED, pos_weight)
        m_imp.fit(X_res_proc, y_rescue)
        imps = m_imp.feature_importances_ if hasattr(m_imp, 'feature_importances_') else (
            np.abs(m_imp.coef_[0]) if hasattr(m_imp, 'coef_') else [0] * len(final_feat_names))

        min_len = min(len(imps), len(final_feat_names))
        pd.DataFrame({"Feature": final_feat_names[:min_len], "Importance": imps[:min_len]}).sort_values("Importance",
                                                                                                        ascending=False).to_excel(
            writer, sheet_name="Importance", index=False)


        if algo_name == 'lr':
            print("📊 正在计算逻辑回归统计学指标 (Statsmodels - 灰色地带)...")
            try:

                X_sm = sm.add_constant(X_res_proc)
                cols_sm = ['const'] + final_feat_names[:X_res_proc.shape[1]]
                logit_model = sm.Logit(y_rescue, X_sm)
                result = logit_model.fit(disp=0, method='bfgs')

                params = result.params
                bse = result.bse
                pvalues = result.pvalues
                wald_chi2 = result.tvalues ** 2
                odds_ratios = np.exp(params)

                conf = result.conf_int()
                conf_exp = np.exp(conf)

                stats_df = pd.DataFrame({
                    "Feature": cols_sm,
                    "Beta (Coef)": params,
                    "S.E.": bse,
                    "Wald (Chi2)": wald_chi2,
                    "P_value": pvalues,
                    "OR": odds_ratios,
                    "OR_95%_Low": conf_exp[:, 0],
                    "OR_95%_High": conf_exp[:, 1]
                })

                stats_df['Significance'] = stats_df['P_value'].apply(
                    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else '')))

                stats_df.to_excel(writer, sheet_name="LR_Statistics_Detail", index=False)
                print("   -> 统计学指标已写入 sheet: LR_Statistics_Detail")

            except Exception as e:
                print(f"⚠️ 统计学指标计算失败: {e}")

    print(f"✅ 文件已保存: {output_path}")

if __name__ == "__main__":
    for algo in ['lr']:
        run_experiment(algo)