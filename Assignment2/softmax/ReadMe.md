# Softmax


## 📦 环境准备

### 1. 检查Python版本
```bash
python --version
# 需要 Python 3.7 或更高版本
```

### 2. 安装依赖
```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn tqdm tensorboard
```

### 3. 验证安装
```python
import torch
import torchvision
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
```

---

## 📖 实验概览

### 学习目标
1. ✅ 手动实现 Softmax 函数
2. ✅ 理解数值稳定性
3. ✅ 进行系统的参数调优实验

