from __future__ import annotations
import numpy as np
import os

# =========================
# utils: dataload & evaluate
# =========================
def load_data(filename: str):
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append([float(v) for v in s.split()])
    A = np.asarray(rows, dtype=np.float64)

    basename = os.path.basename(filename).lower()
    if "train" in basename:
        X, y = A[:, :-1], A[:, -1]
        return X, y
    elif "test" in basename:
        # test_X.txt contains only features, y is not available
        return A, None
    else:
        raise ValueError(f"Unsupported file in load_data: {filename}")

def evaluate_rmse(y_true, y_pred):
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err ** 2))
    return rmse

# =========================
# model: Linear Regression
# =========================
class LinearRegression:
    """
    Multivariate linear regression: y ≈ X @ w + b
    w ∈ R^D, b ∈ R
    """
    def __init__(self, lr, epochs, batch_size, seed):
        self.w = None          # (D,) lazy init when D is known
        self.b = 0.0
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    # ---------- API ----------
    def predict(self, X):
        """Vectorized prediction: X (N,D) -> (N,)"""
        X = np.atleast_2d(X)
        return X @ self.w + self.b

    def fit(self, X, y, solver, verbose: bool = True):
        """
        Train model by:
          - solver="GD"  : mini-batch SGD
          - solver="LSE" : normal equation
        """
        solver = solver.upper()
        if solver == "GD":
            self.train_gd(X, y, verbose=verbose)
        elif solver == "LSE":
            self.train_lse(X, y, verbose=verbose)

    # ---------- solvers ----------
    def train_gd(self, X, y, verbose: bool = True):
        """
        Mini-batch SGD on:
            L = (1/(2m)) * sum (x_i^T w + b - y_i)^2
        Gradients (batch size = m):
            ∇w = X_b^T (X_b w + b - y_b) / m
            ∇b = mean(err)
        """
        X = np.atleast_2d(X)
        y = y.reshape(-1)
        N, D = X.shape

        rng = np.random.RandomState(self.seed)
        if self.w is None or self.w.shape != (D,):
            self.w = rng.randn(D) * 0.001  # small random init

        for ep in range(1, self.epochs + 1):
            idx = np.arange(N)
            rng.shuffle(idx)

            for s in range(0, N, self.batch_size):
                j = idx[s:s + self.batch_size]
                Xb, yb = X[j], y[j]              
                yhat = Xb @ self.w + self.b     
                err = yhat - yb                 
                m = Xb.shape[0]

                # ====================== TODO (students) ======================
                grad_w = (Xb.T @ err) / m
                grad_b = err.mean()

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
                # ====================== END TODO ============================

            if verbose and (ep % max(1, self.epochs // 10) == 0 or ep == 1):
                rmse = evaluate_rmse(y, self.predict(X))
                print(f"[GD][Epoch {ep:4d}] rmse={rmse:.6f}, ||w||={np.linalg.norm(self.w):.4f}, b={self.b:+.4f}")

    def train_lse(self, X, y, verbose: bool = True):
        """
        Closed-form:
            Phi = [X, 1], (Phi^T Phi) theta = Phi^T y, theta=[w; b]
        """
        X = np.atleast_2d(X)
        y = y.reshape(-1)
        N, _ = X.shape

        # ====================== TODO (students) ======================
        Phi = np.concatenate([X, np.ones((N, 1))], axis=1) 
        A = Phi.T @ Phi
        b_vec = Phi.T @ y
        theta = np.linalg.solve(A, b_vec) 
        self.w = theta[:-1]
        self.b = float(theta[-1])
        # ====================== END TODO ============================

        if verbose:
            rmse = evaluate_rmse(y, self.predict(X))
            print(f"[LSE] rmse={rmse:.6f}, ||w||={np.linalg.norm(self.w):.4f}, b={self.b:+.4f}")


# =========================
# main
# =========================
def main():
    # ---- parameter ----
    solver     = "GD"     # "GD" or "LSE"
    lr         = 5e-4     # for GD
    epochs     = 1000      # for GD
    batch_size = 32       # for GD

    train_file = "./input_multi/train.txt"
    test_file  = "./input_multi/test_X.txt"
    out_file_SGD = "./output_multi/SGDpredict.npy"
    out_file_LSE = "./output_multi/LSEpredict.npy"

    # ---- load data ----
    X_train, y_train = load_data(train_file)
    X_test, _  = load_data(test_file)

    # ---- train ----
    model = LinearRegression(lr=lr, epochs=epochs, batch_size=batch_size, seed=42)
    try:
        model.fit(X_train, y_train, solver=solver, verbose=True)
    except NotImplementedError as e:
        raise SystemExit("[ERROR] Please complete the TODO(s) in LinearRegression.train_gd / train_lse.") from e

    # ---- train RMSE ----
    rmse_train = evaluate_rmse(y_train, model.predict(X_train))
    print(f"[Train] RMSE = {rmse_train:.6f}")

    # ---- predict & save ----
    if solver == "GD":
        out_file = out_file_SGD
    else:
        out_file = out_file_LSE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    y_pred = model.predict(X_test).astype(np.float64)
    np.save(out_file, y_pred)
    print(f"[OK] Saved predictions to {out_file}")

if __name__ == "__main__":
    main()
