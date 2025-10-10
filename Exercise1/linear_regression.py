from __future__ import annotations
import numpy as np
import os

# =========================
# utils: dataload & evaluate
# =========================
def load_data(filename: str):
    """
    Load file as:
      - train.txt -> returns (x, y), both shape (N,)
      - test_X.txt -> returns (x, None), x shape (N,)
    """
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals = [float(v) for v in s.split()]
            rows.append(vals)

    A = np.asarray(rows, dtype=np.float64)

    if "train" in os.path.basename(filename).lower():
        if A.shape[1] != 2:
            raise ValueError("train.txt must have exactly 2 columns: x y")
        return A[:, 0], A[:, 1]
    else:
        if A.shape[1] != 1:
            raise ValueError("test_X.txt must have exactly 1 column: x")
        return A[:, 0], None

def evaluate_rmse(y_true, y_pred):
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err ** 2))
    return rmse

# =========================
# model: Linear Regression
# =========================
class LinearRegression:
    def __init__(self, lr, epochs, batch_size, seed):
        # parameters
        rng = np.random.RandomState(seed)
        self.w = float(rng.randn() * 0.001)  # small random init
        self.b = 0.0

        # hyper-parameters (for GD)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def predict(self, x):
        """Vectorized prediction: returns shape (N,)"""
        return self.w * x + self.b

    def fit(self, x, y, solver, lam=0.0, verbose: bool = True):
        """
        Train model by:
          - solver="GD"  : mini-batch SGD
          - solver="LSE" : normal equation
        """
        solver = solver.upper()
        if solver == "GD":
            self.train_gd(x, y, verbose=verbose)
        elif solver == "LSE":
            self.train_lse(x, y, lam=lam, verbose=verbose)
        else:
            raise ValueError("solver must be 'GD' or 'LSE'")

    def train_gd(self, x, y, verbose: bool = True):
        """
        Mini-batch SGD on:
            L = (1/(2m)) * sum_i (w*x_i + b - y_i)^2
        Gradients (batch size = m):
            dL/dw = (1/m) * sum_i (w*x_i + b - y_i) * x_i
            dL/db = (1/m) * sum_i (w*x_i + b - y_i)
        """
        assert x.ndim == 1 and y.ndim == 1 and x.shape[0] == y.shape[0]
        n = x.shape[0]
        rng = np.random.RandomState(self.seed)

        for ep in range(1, self.epochs + 1):
            idx = np.arange(n)
            rng.shuffle(idx)

            for s in range(0, n, self.batch_size):
                j = idx[s:s + self.batch_size]
                xb, yb = x[j], y[j]

                # forward
                y_hat = self.w * xb + self.b
                err = y_hat - yb  # shape (m,)

                # ====================== TODO (students) ======================
                # Compute gradients and do one SGD step:
                grad_w = (err * xb).mean()
                grad_b = err.mean()
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
                # ====================== END TODO ============================

            if verbose and (ep % max(1, self.epochs // 10) == 0 or ep == 1):
                rmse = evaluate_rmse(y, self.predict(x))
                print(f"[GD][Epoch {ep:4d}] rmse={rmse:.6f}, w={self.w:+.4f}, b={self.b:+.4f}")

    def train_lse(self, x, y, lam:float = 0.0,verbose: bool = True):
        """
        Closed-form (normal equation). 
        Build design matrix Phi = [x, 1] of shape (N, 2), and solve:
            (Phi^T Phi) theta = Phi^T y,
        where theta = [w, b]^T and R = diag([1, 0]) so that bias is not penalized.
        """
        assert x.ndim == 1 and y.ndim == 1 and x.shape[0] == y.shape[0]
        Phi = np.stack([x, np.ones_like(x)], axis=1)  # (N, 2)

        # ====================== TODO (students) ======================
        # Implement normal equation with optional ridge:
        R = np.diag([1.0, 0.0])
        A = Phi.T @ Phi + lam * R
        b_vec = Phi.T @ y
        theta = np.linalg.solve(A, b_vec)
        self.w, self.b = float(theta[0]), float(theta[1])
        # ====================== END TODO ============================

        if verbose:
            rmse = evaluate_rmse(y, self.predict(x))
            print(f"[LSE] rmse={rmse:.6f}, w={self.w:+.4f}, b={self.b:+.4f}")


# =========================
# main
# =========================
def main():
    # ---- minimal parameter block ----
    solver = "LSE"          # "GD" or "LSE"
    lr = 5e-4              # for GD
    epochs = 1000           # for GD
    batch_size = 32        # for GD
    train_file = "./input/train.txt"
    test_file = "./input/test_X.txt"
    out_file_SGD = "./output/SGDpredict.npy"
    out_file_LSE = "./output/LSEpredict.npy"

    # ---- load data ----
    x_train, y_train = load_data(train_file)
    x_test, _ = load_data(test_file)

    # ---- train ----
    model = LinearRegression(lr=lr, epochs=epochs, batch_size=batch_size, seed=42)
    try:
        model.fit(x_train, y_train, solver=solver, verbose=True)
    except NotImplementedError as e:
        raise SystemExit("[ERROR] Please complete the TODO(s) in LinearRegression.train_gd / train_lse.") from e

    # ---- train RMSE (sanity check) ----
    rmse_train = evaluate_rmse(y_train, model.predict(x_train))
    print(f"[Train] RMSE = {rmse_train:.6f}")

    # ---- predict & save ----
    if solver == "GD":
        out_file = out_file_SGD
    else:
        out_file = out_file_LSE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    y_pred = model.predict(x_test).astype(np.float64)
    np.save(out_file, y_pred)
    print(f"[OK] Saved predictions to {out_file}")


if __name__ == "__main__":
    main()
