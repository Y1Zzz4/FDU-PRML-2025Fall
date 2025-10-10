import numpy as np
import matplotlib.pyplot as plt

# --- 文件路径定义 ---
TRUTH_PATH = "./input_multi/test_y.txt"
PRED_PATH  = "./output/LSEpredict.npy" # 默认评测LSE的结果
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

def plot_results(X_train, y_train, X_test, y_test, y_pred):
    """
    根据现有数据绘制结果图，样式与原始代码保持一致。
    """
    plt.figure(figsize=(12, 8))

    # 1. 绘制训练和测试数据点
    if X_train is not None and y_train is not None:
        plt.scatter(X_train, y_train, facecolor="none", edgecolor='#e4007f', s=50, label="Train Data")
    if X_test is not None and y_test is not None:
        plt.scatter(X_test, y_test, facecolor="none", edgecolor="r", marker='^', s=50, label="Test Data")

    # 2. 绘制预测线
    # 为了画出平滑的线，需要对X轴排序
    if X_test is not None and y_pred is not None:
        sorted_indices = np.argsort(X_test.flatten())
        plt.plot(X_test[sorted_indices], y_pred[sorted_indices], c='#0075ad', label=f"Prediction")

    plt.legend(fontsize='x-large')
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Label", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    # --- 1. 加载数据并计算RMSE ---
    try:
        y_true = np.loadtxt(TRUTH_PATH, dtype=np.float64).reshape(-1)
        y_pred = load_pred(PRED_PATH)
        print(f"[Eval] RMSE on test data = {rmse(y_true, y_pred):.6f}")
    except FileNotFoundError as e:
        print(f"[Error] Cannot calculate RMSE: {e}. Please check file paths.")
        return # 无法计算RMSE则退出

    # --- 2. 尝试加载用于绘图的数据 ---
    X_test_plot, y_train_plot, X_train_plot = None, None, None
    try:
        X_test = np.loadtxt(TEST_X_PATH, dtype=np.float64)
        if X_test.ndim == 1 and X_test.shape[1] == 1:
            X_test_plot = X_test.flatten()
    except (FileNotFoundError, IOError):
        print(f"[Warning] '{TEST_X_PATH}' not found, skipping test data points in plot.")

    try:
        train_data = np.loadtxt(TRAIN_PATH, dtype=np.float64)
        # 假设最后一列是y，其余是X
        X_train_raw = train_data[:, :-1]
        y_train_plot = train_data[:, -1]
        # 同样只取第一维特征
        if X_train_raw.ndim == 1 and X_train_raw.shape[1] == 1:
            X_train_plot = X_train_raw[:, 0]
    except (FileNotFoundError, IOError):
        print(f"[Warning] '{TRAIN_PATH}' not found, skipping train data points in plot.")
    
    # --- 3. 调用绘图函数 ---
    if X_train_raw.ndim == 1 and X_train_raw.shape[1] == 1:
        plot_results(X_train_plot, y_train_plot, X_test_plot, y_true, y_pred)

if __name__ == "__main__":
    main()
