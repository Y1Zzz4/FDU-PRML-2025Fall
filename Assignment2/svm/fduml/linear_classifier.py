from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from .linear_svm import *


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        使用随机梯度下降训练该线性分类器。

        输入:
        - X: 形状为 (N, D) 的 numpy 数组，包含训练数据；有 N 个训练样本，每个样本的维度为 D。
        - y: 形状为 (N,) 的 numpy 数组，包含训练标签；y[i] = c 表示 X[i] 的标签为 0 <= c < C，其中 C 是类别数。
        - learning_rate: (float) 优化的学习率。
        - reg: (float) 正则化强度。
        - num_iters: (integer) 优化时的迭代步数
        - batch_size: (integer) 每一步使用的训练样本数量。
        - verbose: (boolean) 如果为 true，在优化过程中打印进度。

        输出:
        包含每次训练迭代中损失函数值的列表。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # 假设 y 的取值为 0...K-1，其中 K 是类别数
        if self.W is None:
            # 延迟初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 运行随机梯度下降以优化 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # 从训练数据中采样 batch_size 个元素及其对应的标签，用于本轮梯度下降。        #
            # 将数据存储在 X_batch 中，对应的标签存储在 y_batch 中；采样后 X_batch 的     #
            # 形状应为 (batch_size, dim)，y_batch 的形状应为 (batch_size,)            #
            #                                                                       #
            # 提示: 使用 np.random.choice 生成索引。有放回采样比无放回采样更快。          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 评估损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 执行参数更新
            #########################################################################
            # TODO:                                                                 #
            # 使用梯度和学习率更新权重。                                              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        使用该线性分类器的训练权重预测数据点的标签。

        输入:
        - X: 形状为 (N, D) 的 numpy 数组，包含训练数据；有 N 个训练样本，每个样本的维度为 D。

        返回:
        - y_pred: X 中数据的预测标签。y_pred 是一个长度为 N 的一维数组，每个元素是一个整数，表示预测的类别。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # 实现此方法。将预测标签存储在 y_pred 中。                                    #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数及其导数。
        子类将重写此方法。

        输入:
        - X_batch: 形状为 (N, D) 的 numpy 数组，包含一个包含 N 个数据点的小批量；每个点的维度为 D。
        - y_batch: 形状为 (N,) 的 numpy 数组，包含小批量的标签。
        - reg: (float) 正则化强度。

        返回: 包含以下内容的元组:
        - loss 作为单个浮点数
        - 关于 self.W 的梯度；与 W 形状相同的数组
        """
        pass


class LinearSVM(LinearClassifier):
    """ 使用多类 SVM 损失函数的子类 """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)