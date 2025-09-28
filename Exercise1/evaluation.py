import numpy as np
import matplotlib.pyplot as plt
import os

# --- 文件路径定义 ---
TRUTH_PATH = "./input_multi/test_y.txt"
PRED_PATH  = "./output_multi/LSEpredict.npy" # 默认评测LSE的结果
TEST_X_PATH = "./input_multi/test_X.txt"
TRAIN_PATH = "./input_multi/train.txt"

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差 (RMSE)"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def load_pred(path: str):
    """加载预测结果"""
    return np.load(path).astype(np.float64).reshape(-1)

def main():
    # --- 1. 加载数据并计算RMSE ---
    try:
        y_true = np.loadtxt(TRUTH_PATH, dtype=np.float64).reshape(-1)
        y_pred = load_pred(PRED_PATH)
        print(f"[Eval] RMSE on test data = {rmse(y_true, y_pred):.6f}")
    except FileNotFoundError as e:
        print(f"[Error] Cannot calculate RMSE: {e}. Please check file paths.")
        return # 无法计算RMSE则退出

    # --- 2. 准备绘图数据（统一得到一维横轴） ---
    def as_1d_axis(X):
        X = np.asarray(X)
        if X.ndim == 1:            # 单列文本 -> (N,)
            return X.reshape(-1)
        if X.ndim == 2 and X.shape[1] >= 1:  # 多列 -> 取第0列
            return X[:, 0].reshape(-1)
        return None

    X_test_plot = X_train_plot = y_train_plot = None

    try:
        X_test = np.loadtxt(TEST_X_PATH, dtype=np.float64)
        X_test_plot = as_1d_axis(X_test)
    except (FileNotFoundError, IOError):
        print(f"[Warning] '{TEST_X_PATH}' not found, skipping test points.")

    try:
        train_data = np.loadtxt(TRAIN_PATH, dtype=np.float64)
        y_train_plot = train_data[:, -1].reshape(-1)
        X_train_raw  = train_data[:, :-1]
        X_train_plot = as_1d_axis(X_train_raw)
    except (FileNotFoundError, IOError):
        print(f"[Warning] '{TRAIN_PATH}' not found, skipping train points.")

    plt.figure(figsize=(12, 8))
    if X_train_plot is not None and y_train_plot is not None:
        plt.scatter(X_train_plot, y_train_plot, facecolor="none", edgecolor='#e4007f', s=50, label="Train Data")
    if X_test_plot is not None and y_true is not None:
        plt.scatter(X_test_plot, y_true, facecolor="none", edgecolor="r", marker='^', s=50, label="Test Data")
    if X_test_plot is not None and y_pred is not None:
        order = np.argsort(X_test_plot)
        plt.plot(X_test_plot[order], y_pred[order], c='#0075ad', label="Prediction")

    plt.legend(fontsize='x-large')
    plt.xlabel("Feature", fontsize=12); plt.ylabel("Label", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs("./output_multi", exist_ok=True)
    out_path = "./eval_plot.png"
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    print(f"[Eval] Plot saved to {out_path}")

if __name__ == "__main__":
    main()
