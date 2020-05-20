[TOC]
# Transformer的结构是什么
* Transformer原论文中的结构为Encoder-decoder的结构，像一个Seq-Seq with attention的结构，之后GPT、BERT等模型用到的是Encoder部分。

# Transformer为什么选正余弦函数作为positional Encoding?
* 因为任意位置`$PE_{pos+k}$`可以表示为`$PE_{pos}$`的线性函数，从而可以学习到一些相对位置关系。
* 因为三角公式不受序列长度的限制，也就是可以对 比所遇到序列的更长的序列 进行表示。

# Transformer 中一直强调的 self-attention 是什么？
* **self-attention 的特点在于无视词(token)之间的距离直接计算依赖关系，从而能够学习到序列的内部结构，**

# self-attention 为什么它能发挥如此大的作用
* self-attention 是一种自身和自身相关联的 attention 机制，这样能够得到一个更好的 representation 来表达自身
* 引入 Self Attention 后会更容易捕获句子中长距离的相互依赖的特征

#  关于 self-attention 为什么要使用 Q、K、V，仅仅使用 Q、V/K、V 或者 V 为什么不行
* self-attention 使用 Q、K、V，这样三个参数独立，模型的表达能力和灵活性显然会比只用 Q、V 或者只用 V 要好些，

# self-attention公式中的归一化有什么作用？
* 随`$d_k$`增大，`$q,k$`点积结果也会增大，这样会将softmax推入梯度很小的饱和区域，不利于模型收敛。

# Transformer的优势
* 突破了RNN模型不能并行计算的缺点
* 相比于CNN，计算两个位置之间的依赖所需的操作次数不随距离增长
* 自注意力可以产生更具解释性的模型。

# Transformer、CNN并行优势
* 对于 CNN 和 Transformer 来说，因为它们不存在网络中间状态不同时间步输入的依赖关系，所以可以非常方便及自由地做并行计算改造，这个也好理解。

# Transformer、CNN、RNN时间复杂度对比
* `n`为序列长度，`d`为embedding维度
* Transformer复杂度为`$O(n^2*d)$`
* RNN时间复杂度为`$O(n*d^2)$`
* CNN时间复杂度为`$O(k*n*d^2)$`

![](https://static.leiphone.com/uploads/new/images/20190216/5c67bfc3559b7.jpg?imageView2/2/w/740)

# Transformer中为什么要用LayerNorm而不是BatchNorm
* 如果我们将一批文本组成一个batch，那么BN的操作方向是，对每句话的**第一个词**进行操作。但语言文本的复杂性是很高的，任何一个词都有可能放在初始位置，且词序可能并不影响我们对句子的理解。而BN是针对每个位置进行缩放，这不符合NLP的规律。
* 而LN则是针对一句话进行缩放的，且LN一般用在第三维度，如`[batchsize, seq_len, dims]`中的dims，一般为词向量的维度，或者是RNN的输出维度等等，这一维度各个特征的量纲应该相同。因此也不会遇到上面因为特征的量纲不同而导致的缩放问题。

# Attention适用于哪些任务？
* **长文本任务**，因为长文本本身携带的信息量大，可能会带来信息过载问题，很多任务只需要利用到其中的关键信息，attention可以抽取出这些关键信息
* **涉及到相关的两段文本**，比如说文本匹配以及机器翻译
* 任务很大部分取决于某些特征