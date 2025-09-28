import os, numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

DATA_DIR = "./input_knn"
os.makedirs(DATA_DIR, exist_ok=True)

# ---- 数据规模与难度（可按需微调）----
RANDOM_STATE = 42
N_SAMPLES    = 300      # 200-400
N_CLASSES    = 4        # 4 类
CLUSTER_STD  = 3        # 数值越大，类间越模糊 [2.5, 3, 3.5, 4]
TEST_SIZE    = 0.25
VAL_SIZE     = 0.25

# ---- 生成 2D 多类别数据（适合决策边界可视化）----
X, y = make_blobs(n_samples=N_SAMPLES,
                  centers=N_CLASSES,
                  n_features=2,
                  cluster_std=CLUSTER_STD,
                  random_state=RANDOM_STATE)
X = X.astype(np.float64)
y = y.astype(np.int64)  # 标签为 0..C-1

# ---- 分割：train / val / test ≈ 100 / 50 / 50 ----
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=(TEST_SIZE + VAL_SIZE),
    random_state=RANDOM_STATE, stratify=y
)
rel_test = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=rel_test,
    random_state=RANDOM_STATE, stratify=y_rest
)

# ---- 保存 ----
for name, arr in [
    ("X_train.npy", X_train), ("y_train.npy", y_train),
    ("X_val.npy",   X_val),   ("y_val.npy",   y_val),
    ("X_test.npy",  X_test),  ("y_test.npy",  y_test),
]:
    np.save(os.path.join(DATA_DIR, name), arr)

print(f"[OK] Saved to {DATA_DIR}")
print(f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}, D={X_train.shape[1]}, classes={len(np.unique(y))}")


REQUIRED_FILES = [
    "X_train.npy", "y_train.npy",
    "X_val.npy",   "y_val.npy",
    "X_test.npy",  "y_test.npy",
]

def load_prepared_dataset(data_dir: str = DATA_DIR):
    """
    从 data_dir 读取预先保存的 KNN 数据集（.npy），并做最小必要校验。
    期望文件：
      X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
    返回：X_train, y_train, X_val, y_val, X_test, y_test
    """
    paths = {name: os.path.join(data_dir, name) for name in REQUIRED_FILES}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    X_train = np.load(paths["X_train.npy"]).astype(np.float64)
    y_train = np.load(paths["y_train.npy"]).astype(int).reshape(-1)

    X_val   = np.load(paths["X_val.npy"]).astype(np.float64)
    y_val   = np.load(paths["y_val.npy"]).astype(int).reshape(-1)

    X_test  = np.load(paths["X_test.npy"]).astype(np.float64)
    y_test  = np.load(paths["y_test.npy"]).astype(int).reshape(-1)

    # 基本一致性检查（简洁为主）
    assert X_train.shape[0] == y_train.shape[0], "train size mismatch"
    assert X_val.shape[0]   == y_val.shape[0],   "val size mismatch"
    assert X_test.shape[0]  == y_test.shape[0],  "test size mismatch"
    D = X_train.shape[1]
    assert X_val.shape[1] == D and X_test.shape[1] == D, "feature dim mismatch across splits"

    print(f"[Load] train={X_train.shape[0]} | val={X_val.shape[0]} | test={X_test.shape[0]} | D={D}")
    return X_train, y_train, X_val, y_val, X_test, y_test
