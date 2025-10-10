import numpy as np
import accuracy_error as acc_err      # accuracy / MSE
import evaluation_metrics as eval        # precision / recall / f1
from datetime import datetime

# ============== 学生信息（请填写） ==============
STUDENT_NAME = "吴卓阳"   # 例如：张三
STUDENT_ID   = "23307130392"   # 例如：2025123456
# ==============================================

def test_basic():
    print("[BASIC] accuracy / MSE")
    # accuracy: 3/4 = 0.75
    yt = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0])
    yp = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
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
    print("[PRF] precision / recall / f1 (binary)")
    # y_true=[1,1,0,0], y_pred=[1,0,1,0]
    # tp=1, fp=1, fn=1, tn=1 -> P=0.5, R=0.5, F1=0.5
    ytb = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0])
    ypb = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
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

def _print_header():
    name, sid = STUDENT_NAME, STUDENT_ID
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("实验：Part 1 - 分类评测指标")
    print(f"姓名：{name}    学号：{sid}")
    print(f"时间：{ts}")
    print("=" * 60)
    return name, sid

if __name__ == "__main__":
    _print_header()
    test_basic()
    test_eval()
