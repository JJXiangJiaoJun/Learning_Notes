网络选择思路：加入Attention结构，提高分类效果
[TOC]
## （1）网路关键点
1. **Attention Module**
    <br/>作者在Residual block中产生了两个分支，一个分支用来生成feature map，另一个分支用来生成attention map,最后输出为 attention map*feature map,这样可以使feature map中重要的特征权重变大，从而抑制一些噪声
2. **改进**
    * 数据增广
    <br/>使用cutout，随机擦除一小块区域，增加有遮挡物体的分类精度,Random crop..等
    * 训练tricks
        * mix up train
        * label smoothing
        * No bias Decay:Weight Decay只用于weight，不用于bias,BN的 `$\gamma,\beta$`
        * LR warmup 和 LR cosine decay
        * Deep mutual Learning
    * 预测trick
        * 多个模型进行ensemble