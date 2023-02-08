# 线性判别分析LDA

## 基本思想

$LDA$的思想非常朴素，给定训练样例集，设法将样例投影到一条直线上，使得同类样例的**投影点**尽可能接近、异类样例点尽可能远离；在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定样本的类别。

## 损失函数

使得同类投影点协方差尽可能小，类中心之间的距离尽可能大，即最大化$J$：
$$
J=\frac{||\omega^T\mu_0-\omega^T\mu_1||_2^2}{\omega^T\Sigma_0\omega+\omega^T\Sigma_1\omega}\\
=\frac{\omega^T(\mu_0-\mu_1)(\mu_0-\mu_1)^T\omega}{\omega^T(\Sigma_0+\Sigma_1)\omega}
$$
其中，分母是同类投影点的协方差，分子是类中心之间的距离。

定义类内散度矩阵$S_w=\Sigma_0+\Sigma_1$与类间散度矩阵$S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T$
$$
J=\frac{\omega^TS_b\omega}{\omega^TS_w\omega}
$$
