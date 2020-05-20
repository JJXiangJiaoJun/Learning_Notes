[TOC]
# 为什么NLP网络中经常使用Adam优化器以及存在的问题
## NLP语言模型的特点
通常自然语言处理模型的输入是非常稀疏的。对于包含几十万上百万词的词表，在训练的每个 Batch 中能出现的独立词数不超过几万个。也就是说，在每一轮梯度计算过程中，只有几万个词的 embedding 的梯度是非 0 的，其它 embedding 的梯度都是 0。

## Adam优化器的特点

Adam 优化器可以说是目前使用最广泛、收敛速度较快且收敛过程较稳定的优化器。Adam 的计算公式如图所示。可以看到公式中梯度的计算使用了动量原理，每一轮用于梯度下降的梯度是当前计算的真实梯度与上一轮用于梯度下降的梯度的加权和。这样动量的引入可以防止训练时产生震荡。Adam 优化器的学习率对于不同参数也是不同的，由该参数历史每一轮的真实梯度的大小决定。好处是对于 NLP 这种输入极度稀疏且输入特征极度不平衡（例如整个预料库中“我”这样的词经常出现，而“拉姆塞”这样的词只出现几次）的任务，学习率是自适应的，一些在一次训练 epoch 中只更新几次的 embedding，在训练后期还是会有较大的学习率。

![](https://upload-images.jianshu.io/upload_images/12224193-b6737cab31519d71.png?imageMogr2/auto-orient/strip|imageView2/2/w/245/format/webp)

## 问题
NLP 输入稀疏的特点与 Adam 使用动量计算梯度的特点相结合就引入了麻烦。每一轮更新参数时，只有极少数 embedding 的梯度是非 0 的，大部分 embedding 的梯度是 0 即上图公式中的 gt 是 0。但是，计算了动量之后，这些原本梯度都应该是 0 的 embedding 有了非零梯度 mt 用于梯度下降更新。想象一个极端的例子，“拉姆塞”这个词在一个 epoch 中只在第一个 batch 出现了，于是第一个 batch 计算了“拉姆塞”这个 embedding 的真实梯度 g0 用于更新参数，在以后的每个 batch 中虽然“拉姆塞”这个词没有出现过，Adam 都会计算它的动量梯度 mt，并用于更新“拉姆塞”这个 embedding，实际上方向与 g0 完全相同，只是每一轮做一次 β1 倍的衰减。这样的做法就相当于对这些出现次数较少的低频词的 embedding，每次梯度下降的等效学习率是非常大的，容易引起类似过拟合的问题。

## 解决办法
知道了问题的根节，解决方法就很简单了，每轮迭代只更新这个 batch 中出现过的词的 embedding 即可。TensorFlow 中可以使用 tf.contrib.opt.LazyAdamOptimizer，也可参考 https://www.zhihu.com/question/265357659/answer/580469438 的实现。

# BERT的基本原理
* BERT是`Bidirectional Encoder Representation from Transformers`的缩写，整体上是一个自编码语言模型，是由transformer基本结构搭建而成，Pre-train时设计了两个任务进行优化
    * Masked Language Model，随机将一句话中的某些词进行Mask，并基于上下文预测被Mask的词
    * Next Sentence，预测输入的两个句子是否连续，引入这个的目的是为了让模型更好地学到文本片段之间的关系
* BERT相对于原来的RNN、LSTM结构可以做到并行执行，同时提取在句子中的关系特征，并且能在多个不同层次提取关系特征，进而反映句子语义，相对于word2Vec，BERT能根据句子上下文获取词义，解决多义词问题，BERT由于模型参数大，也存在容易过拟合的问题。

# BERT为什么要取消NSP任务
* NSP任务其实不仅仅包含了句间关系预测，也包含了主题预测，比如说一句话来自新闻主题，一句话来自医疗主题，模型会倾向于通过主题去预测，这样任务是比较简单的，所以一般会改成其他任务，比如预测两个句子顺序是否交换

# 为什么BERT比ELMo效果好？ELMo和BERT的区别是什么？
## 为什么BERT比ELMo效果好
* 从网络结构和最后的实验效果看，BERT比ELMo效果好主要原因有三点：
    * LSTM抽取特征的能力远弱于Transformer
    * 双向拼接特征融合能力偏弱
    * BERT的训练数据以及模型参数远远多于ELMo
## ELMo和BERT的区别是什么
* ELMo通过语言模型得到句子中的词语的动态embedding，然后将此作为下游任务的补充输入，是一种Featrue-based方式，而BERT是基于Pre-train Finetune方式，使用Pre-train参数初始化Fine-tune的模型结构。

# BERT有什么局限性
* BERT在Pre-train阶段，假设句子中多个单次被Mask，这些被Mask掉的单词之间没有任何关系，是条件独立的
* BERT在预训练阶段会出现特殊的[MASK]，而fine-tune阶段没有，这就出现了Pre-train和Fine-tune不一致的问题。

# BERT输入和输出分别是什么？
* **输入**
    * BERT输入有`input_ids,segment_ids,input_masks`，之后会在输入端生成原始词向量，位置向量，以及文本向量，一起相加作为网络的输入
* **输出**
    * BERT可以调用`get_pooled_ouput`获得全局的语义表征向量
    * `get_sequence_output`则是获得各个词的表征向量
# BERT模型为什么要用Mask？如何做Mask？Mask和CBOW有什么异同点
## BERT模型为什么要用mask
* BERT通过随机mask句子中的一部分词，然后根据上下文预测这些被mask得此，这个是Denosing AutoEncoder做法，加入Mask相当于在输入端加入噪声，这种方式优点是能够比较自然的融入双向语言模型，缺点是引入了噪声，导致Pre-train和Fine-tune不一致。

## 相对于CBOW有什么异同点
* **相同点**
    * 核心思想都是通过上下文来预测中心词
* **不同点**
    * CBOW中每个词都会成为中心词，而BERT只有Mask掉的词会成为中心词
    * CBOW的输入只有带预测的上下文，而BERT输入中带有mask token
    * CBOW训练结束之后，每个词向量都是唯一的，而BERT输入的词在不同上下文环境中，最后输出的表征向量也是不一样的。

# 针对中文BERT有什么能改进的地方
* 使用词粒度

# attention类型
* 计算区域
    * soft Attention
    * hard Attention
    * Local Attentin
* 所用信息
    * General Attention
    * Local Attention
# attention有什么优点和缺点
## 优点
* **参数少**，模型复杂度相对于CNN、RNN比，复杂度更小，参数也更少
* **速度快**，Attention机制每一步计算不依赖于上一时间步的计算结果，相对于RNN可以很容易的做到并行
* **效果好**，Attention是挑重点，可以同时捕获局部和全局的信息，当输入一段长文字时，可以抓住其中的重点，不丢失信息。
## 缺点
* 没法捕捉位置信息，即没法学习序列中的顺序关系。这点可以通过加入位置信息，如通过位置向量来改善，具体可以参考最近大火的BERT模型。

# 为什么要使用多头注意力
* 多头注意力本质上是多个attention的计算，然后做一个集成作用
* 和CNN使用多个通道卷积效果类似
* 论文中说到这样的好处是可以允许模型在不同的表示子空间里学习到相关的信息，后面还会根据attention可视化来验证。
* 当你浏览网页的时候，你可能在颜色方面更加关注深色的文字，而在字体方面会去注意大的、粗体的文字。这里的颜色和字体就是两个不同的表示子空间。

# Layer Normalization
 * Batch Norm是在batch上，对NHW做归一化，就是对每个单一通道输入进行归一化，这样做对小batchsize效果不好；
 * Layer Norm在通道方向上，对CHW归一化，就是对每个深度上的输入进行归一化，主要对RNN作用明显；
    * LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；
    * BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差。
    * LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。
* Instance Norm在图像像素上，对HW做归一化，对一个图像的长宽即对一个像素进行归一化，用在风格化迁移；
* Group Norm将channel分组，有点类似于LN，只是GN把channel也进行了划分，细化，然后再做归一化
* Switchable Norm是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

# Glue（高斯误差线性单元）
* 相比Relu：Relu将小于0的数据映射到0，将大于0的给与  等于 映射操作，虽然性能比sigmoid好，但是缺乏数据的统计特性，而Gelu则在relu的基础上加入了统计的特性。论文中提到在好几个深度学习任务中都优于Relu的效果。

# AdamWeightDecayOptimizer
* 引入了正则项的Adam优化器

# RoBERTa、ALBERT、ERNIE比较
* **RoBERTa**
    * 动态Masking
    * 移除next predict loss，采用了连续的full-sentences作为输入
    * 更大batch_size
    * 训练数据增大
* **ALBERT**
    * **factorized embeddng parameterization（嵌入向量的参数分解）**. 他们做的第一个改进是针对于Vocabulary Embedding。现将one-hot编码映射到一个低维词嵌入空间，之后再映射到隐藏空间
    * **跨层参数共享**， ALBERT 采用的是贡献所有层的所有参数。
    * **SOP（句间连贯性损失）**

# BERT各种蒸馏方式
* **从transformer到非transformer框架的蒸馏**
    * 使用教师模型的输出层作为教师信号
* **从transformer到transformer框架的蒸馏**
    * 可以不仅仅蒸馏输出结果，对中间层也可以进行蒸馏，tinyBERT中就对embedding，attention，hidden，输出层都进行了蒸馏 
    * 两阶段蒸馏，
        * 首先在通用大数据上，对Pre-train模型进行蒸馏，得到通用模型，
        * 然后再在对应下游任务上fine-tune后再进行一次蒸馏

# BERT论文中各种BERT的参数
* BERT_base(L=12,H=768,A=12,Total Parameters=110M)
* BERT_Large(L=12,H1024,A=16,Total Parameters=340M)

# BERT双向体现在哪里
* BERT语言模型Pre-train的时候，self-attention使用的是双向，同时与上下文进行attention

# 自回归和自编码
## 自回归
* 根据上文内容预测下文，或者根据下文内容预测上文，是基于概率论的乘法公式分解，GPT，ELMo、XLNet本质上都是自回归语言模型
    * **优点**：和下游NLP任务相关，比如生成式NLP任务，文本摘要，机器翻译等，实际生成内容时，就是从左到右的，这样Pre-train和finetune阶段任务一直
    * **缺点**：只能利用单向信息，不能同时利用双向信息融合。
## 自编码
* 用一个神经网络把输入映射为一个低维特征（通常输入会加一些噪声），这就是编码部分，之后再尝试将这个特征进行还原。BERT就是自编码模型
    * **优点**：能够比较自然的融入双向语言模型
    * **缺点**：引入了噪声，存在Pre-train和Fine-tune两阶段不一致的问题。