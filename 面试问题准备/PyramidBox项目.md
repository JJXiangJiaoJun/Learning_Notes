网络选择思路：提高小人脸的检测比例
[TOC]

## （1） 网络关键点
1. **Low-level-FPN**<br/>
    和FPN类似，但是最后两层特征图没有包含在里面，因为最高两层特征感受野太大，反而会引入噪声信息，影响小人脸的检测
2. **CPM contextual-sensetive prediction Module**
    <br/>类似于Inception，有多个并行的模块，各个分支的卷积大小步幅不同，这样输出特征图的感受野不同，最后将Feature map concat起来，预测的时候使用了Max-in-out，增加小人脸Recall,并减小FP
3.  **半监督学习  Pyramid Anchor**
    <br/>对人脸额外加入了头部和身体的标签，自动生成，不需要人工标注
4.  **Data-anchor-sampling**
    <br/>该方法通过reshape一个图片的随机人脸到一个随机的更小锚框尺度。
    * 首先随机选择一个尺度为`$S_{face}$`的人脸
    * 从anchor中找到最接近gt的anchor尺度
    * `$i_{anchor} = argmin (s_{anchor_i}-s_{face})$`
    * 从`{0，1，...，min(5,i_{anchor}+1)}`中随机选择一个编号`$i_{target}$`
    * 最后 `$S_{target}=random(S_{itarget}/2,S_{itarget}/2*2)$`
    * 图像缩放比 `$S^*=S_{target}/S_{face}$`

    (1)较小人脸的比例高于较大人脸
    (2)通过较大的人脸生成较小的人脸，提高样本在小尺度上的多样性

## （2）改进
1. **数据分析处理**
    <br/>数据由5个不同场景组成，内容差异大，数量也差异较大。我们对数据量少的数据进行增广（加噪点，模糊，修改色度，旋转平移，仿射变换，高斯模糊，RGB通道随机交换），使其数量均衡

2. **训练tricks**
    * val划分和测试集数据分布基本保持一致，保持在val在mAP和test上尽量一致，方便调参
    * LR采用了warm up和consine decay方式
    * 对不同数据集训练单一模型，分别预测
3. **测试trick**
    * 图像金字塔测试
    * 翻转测试
    * 原图测试
    * 放大两倍测试
    * bbox_vote(选取IoU大的加权平均)
    * predict head 使用DenseNet结构
    * 用通用模型和专用模型分别进行预测ensemble
