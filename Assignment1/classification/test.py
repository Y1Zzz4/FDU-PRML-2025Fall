import numpy as np
import accuracy_error_groundtruth as acc_err      # accuracy / MSE
import evaluation_metrics_grountruth as eval        # precision / recall / f1

def test_basic():
    print("[BASIC] accuracy / MSE")
    # accuracy: 3/4 = 0.75
    yt = np.array([0, 1, 1, 0])
    yp = np.array([0, 1, 0, 0])
    try:
        acc = acc_err.accuracy_score(yt, yp)
        print(f" - accuracy expected 0.75 -> got {acc:.2f} : {'PASS' if abs(acc-0.75)<1e-8 else 'FAIL'}")
    except NotImplementedError:
        print(" - accuracy NOT IMPLEMENTED")

    # mse: (0-0)^2 + (2-1)^2 + (1-2)^2 = 2 -> /3 = 0.6667
    yt_r = np.array([0., 1., 2.])
    yp_r = np.array([0., 2., 1.])
    try:
        mse = acc_err.mean_squared_error(yt_r, yp_r)
        print(f" - mse      expected 0.6667 -> got {mse:.4f} : {'PASS' if abs(mse-2/3)<1e-6 else 'FAIL'}")
    except NotImplementedError:
        print(" - mse NOT IMPLEMENTED")

def test_eval():
    print("\n[PRF] precision / recall / f1 (binary)")
    # y_true=[1,1,0,0], y_pred=[1,0,1,0]
    # tp=1, fp=1, fn=1, tn=1 -> P=0.5, R=0.5, F1=0.5
    ytb = np.array([1, 1, 0, 0])
    ypb = np.array([1, 0, 1, 0])
    try:
        p = eval.precision_score(ytb, ypb)
        r = eval.recall_score(ytb, ypb)
        f = eval.f1_score(ytb, ypb)
        ok_p = abs(p - 0.5) < 1e-8
        ok_r = abs(r - 0.5) < 1e-8
        ok_f = abs(f - 0.5) < 1e-8
        print(f" - precision expected 0.50 -> got {p:.2f} : {'PASS' if ok_p else 'FAIL'}")
        print(f" - recall    expected 0.50 -> got {r:.2f} : {'PASS' if ok_r else 'FAIL'}")
        print(f" - f1        expected 0.50 -> got {f:.2f} : {'PASS' if ok_f else 'FAIL'}")
    except NotImplementedError:
        print(" - PRF NOT IMPLEMENTED")

if __name__ == "__main__":
    test_basic()
    test_eval()
