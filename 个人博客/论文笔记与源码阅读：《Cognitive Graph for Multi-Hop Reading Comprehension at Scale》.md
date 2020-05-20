@[TOC](目录)
论文链接如下
> Cognitive Graph for Multi-Hop Reading Comprehension at Scale
[论文链接](https://arxiv.org/abs/1905.05460?context=cs.CL)

# 总结
&emsp;&emsp;传统的MRC一般分为四个任务，完形填空（Cloze Test）、选择题（Multiple Choice）、文本抽取（Span Extraction）、开放式问答（Free Answering）。然而这些传统的MRC方法都有一个缺点，**输出answer只能从当前输入passage推理得到，难以引入外部知识，进行多跳的复杂推理**。这篇论文给了我耳目一新的感觉，与之前MRC不同的是，论文中整个系统的输出不仅仅是一个答案，**而是一个带有答案、实体节点的graph**，通过这种方式，不仅可以进行多跳的推理从而得到答案，而且可以获得整个推理的路径，这篇论文结合了NLP大杀器BERT和GCN，我觉得这种Multi-hop MRC应该是以后发展的方向，实用性很高。


# 论文笔记
## Introduction
深度学习出现以来，攻克了MRC中许多问题，但是机器阅读理解和人类的阅读理解还有三个方面存在差距：
1. Reasoning ability。单段落的QA网络一般是从输入的passage找到匹配问题的句子作为答案输出，但是这种方式难以处理需要复杂推理的回答，因此Multi-hop QA则是我们要攻克的难点
2. Explainability。很多时候我们是直接抽取出答案进行回答，并没有展现我们的推理过程
3. Scalability。

CogQA正是针对上面三点进行设计，CogQA有两个系统，其中各个系统的功能如下:
* 系统一：BERT组成，作用是来抽取关键信息，比如说**下一跳的实体节点、当前可能的答案节点**
* 系统二：GCN，用来做推理，从系统一的输入建立一张图，并在这张图上进行推理，得到最终的答案

CogQA的系统结构图如下图所示：

![cogQA结构图](https://img-blog.csdnimg.cn/20200501173920172.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hpYW5nSmlhb0p1bl8=,size_16,color_FFFFFF,t_70#pic_center)
# Cognitve Graph QA Framework
论文中很多细节只有看了代码才能真正理解清楚，下面我结合论文和我自己对代码的思考来说说我对这篇论文的理解。

## 输入数据的格式
我们首先来看看源码中，训练数据的格式，我节选了其中一条数据
```json

    {
        "supporting_facts": [
            [
                "Arthur's Magazine",
                0,
                []
            ],
            [
                "First for Women",
                0,
                []
            ]
        ],
        "level": "medium",
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "context": [
            [
                "Radio City (Indian radio station)",
                [
                    "Radio City is India's first private FM radio station and was started on 3 July 2001.",
                    " It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).",
                    " It plays Hindi, English and regional songs.",
                    " It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.",
                    " Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.",
                    " The Radio station currently plays a mix of Hindi and Regional music.",
                    " Abraham Thomas is the CEO of the company."
                ]
            ],
            [
                "History of Albanian football",
                [
                    "Football in Albania existed before the Albanian Football Federation (FSHF) was created.",
                    " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .",
                    " Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.",
                    " In 1932, Albania joined FIFA (during the 12\u201316 June convention ) And in 1954 she was one of the founding members of UEFA."
                ]
            ],
            [
                "Echosmith",
                [
                    "Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.",
                    " Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.",
                    " Echosmith started first as \"Ready Set Go!\"",
                    " until they signed to Warner Bros.",
                    " Records in May 2012.",
                    " They are best known for their hit song \"Cool Kids\", which reached number 13 on the \"Billboard\" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.",
                    " The song was Warner Bros.",
                    " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.",
                    " The band's debut album, \"Talking Dreams\", was released on October 8, 2013."
                ]
            ],
            [
                "Women's colleges in the Southern United States",
                [
                    "Women's colleges in the Southern United States refers to undergraduate, bachelor's degree\u2013granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.",
                    " Many started first as girls' seminaries or academies.",
                    " Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.",
                    " Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level."
                ]
            ],
            [
                "First Arthur County Courthouse and Jail",
                [
                    "The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum."
                ]
            ],
            [
                "Arthur's Magazine",
                [
                    "Arthur's Magazine (1844\u20131846) was an American literary periodical published in Philadelphia in the 19th century.",
                    " Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.",
                    " In May 1846 it was merged into \"Godey's Lady's Book\"."
                ]
            ],
            [
                "2014\u201315 Ukrainian Hockey Championship",
                [
                    "The 2014\u201315 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.",
                    " Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.",
                    " Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.",
                    " The regular season included just 12 rounds, where all the teams went to the semifinals.",
                    " In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk."
                ]
            ],
            [
                "First for Women",
                [
                    "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
                    " The magazine was started in 1989.",
                    " It is based in Englewood Cliffs, New Jersey.",
                    " In 2011 the circulation of the magazine was 1,310,696 copies."
                ]
            ],
            [
                "Freeway Complex Fire",
                [
                    "The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.",
                    " The fire started as two separate fires on November 15, 2008.",
                    " The \"Freeway Fire\" started first shortly after 9am with the \"Landfill Fire\" igniting approximately 2 hours later.",
                    " These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda."
                ]
            ],
            [
                "William Rast",
                [
                    "William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.",
                    " It is most known for their premium jeans.",
                    " On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.",
                    " The label also produces other clothing items such as jackets and tops.",
                    " The company started first as a denim line, later evolving into a men\u2019s and women\u2019s clothing line."
                ]
            ]
        ],
        "answer": "Arthur's Magazine",
        "_id": "5a7a06935542990198eaf050",
        "type": "comparison",
        "Q_edge": [
            [
                "Arthur's Magazine",
                "Arthur's Magazine",
                33,
                50
            ],
            [
                "First for Women",
                "First for Women",
                54,
                69
            ]
        ]
    },
```
比较重要的字段说明如下：
|  字段| 描述|
|--|--|
|supporting_facts  | 里面是一组列表，每个数据第一个是实体名，第二个是该实体出现的起始位置 ，然后是该实体的下一跳实体(Next hop)|
|question|问题|
|context|一共有十条数据，每条数据都包含一个实体，以及一段描述文字，这个就是之后输入的Para[x]
|answer|答案|

# System 1
系统一的框图如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200501180556456.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hpYW5nSmlhb0p1bl8=,size_16,color_FFFFFF,t_70#pic_center)

我们首先总体的来看一下系统一的结构：
## 系统一输入
* 标准的BERT输入，**[CLS] + Question + [SEP] + clues[x,G] + [SEP] + Para[x]**，我们来分别看看每一个部分都是对应于输入数据中的哪些地方
	1. `Question`，这个很简单，就是对应于输入中的`Question`
	2. `clues[x,G]`，论文中这里定义的是`x`的**线索**，**也就是`x`的上一跳节点中包含实体`x`的那些句子**,举个例子，比如说有个context如下
	 > created by Matt Groening who named the character after President Richard Nixon's middle name.

	假设当前句子可以推理出下一跳实体节点为` President Richard Nixon'`，那么这个句子就是` President Richard Nixon'`的线索。对应于输入数据，那么就是`supporting_facts`中的数据
	3. `Paragraph[x]`，这个的话对应于输入数据中的`context`，context中有十个实体，每个实体都有自己对应的`Paragraph[x]`

## 系统一输出
* 系统一输出类似于PointNet中的输出，会分别预测`Hop span`以及`Ans span`的开始和结束位置，也就是对每个token，计算它作为起始位置和结束位置的概率，从而得到当前答案节点和下一跳的实体节点。
* 系统一还会输出`sem[x,Q,clues]`,也就是当前输入的隐层语义表示，用来作为新生成的下一跳实体节点和答案节点的隐层表示,

## 系统一源码
系统一的源码在 `model.py`中，为`BertForMultiHopQuestionAnswering`类，其关键代码如下：
```python
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        sep_positions=None,
        hop_start_weights=None,
        hop_end_weights=None,
        ans_start_weights=None,
        ans_end_weights=None,
        B_starts=None,
        allow_limit=(0, 0),
    ):
        """ Extract spans by System 1.
        
        Args:
            input_ids (LongTensor): Token ids of word-pieces. (batch_size * max_length)
            token_type_ids (LongTensor): The A/B Segmentation in BERTs. (batch_size * max_length)
            attention_mask (LongTensor): Indicating whether the position is a token or padding. (batch_size * max_length)
            sep_positions (LongTensor): Positions of [SEP] tokens, mainly used in finding the num_sen of supporing facts. (batch_size * max_seps)
            hop_start_weights (Tensor): The ground truth of the probability of hop start positions. The weight of sample has been added on the ground truth. 
                (You can verify it by examining the gradient of binary cross entropy.)
            hop_end_weights ([Tensor]): The ground truth of the probability of hop end positions.
            ans_start_weights ([Tensor]): The ground truth of the probability of ans start positions.
            ans_end_weights ([Tensor]): The ground truth of the probability of ans end positions.
            B_starts (LongTensor): Start positions of sentence B.
            allow_limit (tuple, optional): An Offset for negative threshold. Defaults to (0, 0).
        
        Returns:
            [type]: [description]
        """
        batch_size = input_ids.size()[0]
        device = input_ids.get_device() if input_ids.is_cuda else torch.device("cpu")
        sequence_output, hidden_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        semantics = hidden_output[:, 0] #获取sem[x,Q,clues]作为隐层的语义表示
        
        # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        if sep_positions is None:
            return semantics  # Only semantics, used in bundle forward
        else:
            max_sep = sep_positions.size()[-1]
        if max_sep == 0:
            empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return (
                empty,
                empty,
                semantics,
                empty,
            )  # Only semantics, used in eval, the same ``empty'' variable is a mistake in general cases but simple

        # Predict spans
        logits = self.qa_outputs(sequence_output)
        hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(
            1, dim=-1
        )
		#计算答案抽取的loss
        hop_start_logits = hop_start_logits.squeeze(-1)
        hop_end_logits = hop_end_logits.squeeze(-1)
        ans_start_logits = ans_start_logits.squeeze(-1)
        ans_end_logits = ans_end_logits.squeeze(-1)  # Shape: [batch_size, max_length]

        if hop_start_weights is not None:  # Train mode
            lgsf = torch.nn.LogSoftmax(
                dim=1
            )  # If there is no targeted span in the sentence, start_weights = end_weights = 0(vec)
            hop_start_loss = -torch.sum(
                hop_start_weights * lgsf(hop_start_logits), dim=-1
            )
            hop_end_loss = -torch.sum(hop_end_weights * lgsf(hop_end_logits), dim=-1)
            ans_start_loss = -torch.sum(
                ans_start_weights * lgsf(ans_start_logits), dim=-1
            )
            ans_end_loss = -torch.sum(ans_end_weights * lgsf(ans_end_logits), dim=-1)
            hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
            ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
        else:
            # In eval mode, find the exact top K spans.
            K_hop, K_ans = 3, 1
            hop_preds = torch.zeros(
                batch_size, K_hop, 3, dtype=torch.long, device=device
            )  # (start, end, sen_num)
            ans_preds = torch.zeros(
                batch_size, K_ans, 3, dtype=torch.long, device=device
            )
            ans_start_gap = torch.zeros(batch_size, device=device)
            for u, (start_logits, end_logits, preds, K, allow) in enumerate(
                (
                    (
                        hop_start_logits,
                        hop_end_logits,
                        hop_preds,
                        K_hop,
                        allow_limit[0],
                    ),
                    (
                        ans_start_logits,
                        ans_end_logits,
                        ans_preds,
                        K_ans,
                        allow_limit[1],
                    ),
                )
            ):
                for i in range(batch_size):
                    if sep_positions[i, 0] > 0:
                        values, indices = start_logits[i, B_starts[i] :].topk(K)
                        for k, index in enumerate(indices):
                            if values[k] <= start_logits[i, 0] - allow:  # not golden
                                if u == 1: # For ans spans
                                    ans_start_gap[i] = start_logits[i, 0] - values[k]
                                break
                            start = index + B_starts[i]
                            # find ending
                            for j, ending in enumerate(sep_positions[i]):
                                if ending > start or ending <= 0:
                                    break
                            if ending <= start:
                                break
                            ending = min(ending, start + 10)
                            end = torch.argmax(end_logits[i, start:ending]) + start
                            preds[i, k, 0] = start
                            preds[i, k, 1] = end
                            preds[i, k, 2] = j
        return (
            (hop_loss, ans_loss, semantics)
            if hop_start_weights is not None
            else (hop_preds, ans_preds, semantics, ans_start_gap)
        )
```
### 前向运算过程
forward方法则是整个系统一的**前向运算步骤**：
* 将`input_ids, token_type_ids, attention_mask` 获得，`sequence_output, hidden_output`，其中`sequence_output`用来预测`next hop`以及`ans node`,类似于MRC中的答案抽取，而`hidden_output`在源码中为BERT倒数第4层的输出，作为`sem[x,Q,clues]`的隐层表示。
* 如果是`Train Mode`，那么将会直接计算`span extraction loss`,主要是`hop_loss` 和`ans_loss`,本质上是交叉熵损失
* 如果是`Eval mode`，那么将会预测`top K`个span，其中`K_hop = 3 , K_ans = 1`,即分别预测三个`hop span`和一个`ans_node`，从源码中可以看出，`[CLS]`位置输出的logits为阈值，只有大于这个值的节点才会输出
### 代码细节
简单的对代码的细节分析一下：
1. 输入参数部分
```python
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        sep_positions=None,
        hop_start_weights=None,
        hop_end_weights=None,
        ans_start_weights=None,
        ans_end_weights=None,
        B_starts=None,
        allow_limit=(0, 0),
    ):
```
其中 `input_ids、token_type_ids、attention_mask`为标准的BERT输入，`sep_position`为`[SEP]`分隔符的位置用来确定`supporting fact`句子个数,`attention_mask`用来标记当前字符是否为填充,`hop_start_weights、hop_end_weights、ans_start_weights、ans_end_weights`为训练时使用的标签，用来计算交叉熵损失，`B_starts`是Sentence B的起始位置，也就是`Para[x]`的起始位置，进行Span predict时需要用上

2.  训练前向运算部分
```python
        batch_size = input_ids.size()[0]
        device = input_ids.get_device() if input_ids.is_cuda else torch.device("cpu")
        sequence_output, hidden_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        semantics = hidden_output[:, 0]
        # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        if sep_positions is None:
            return semantics  # Only semantics, used in bundle forward
        else:
            max_sep = sep_positions.size()[-1]
        if max_sep == 0:
            empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return (
                empty,
                empty,
                semantics,
                empty,
            )  # Only semantics, used in eval, the same ``empty'' variable is a mistake in general cases but simple

        # Predict spans
        logits = self.qa_outputs(sequence_output)
        hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(
            1, dim=-1
        )
        hop_start_logits = hop_start_logits.squeeze(-1)
        hop_end_logits = hop_end_logits.squeeze(-1)
        ans_start_logits = ans_start_logits.squeeze(-1)
        ans_end_logits = ans_end_logits.squeeze(-1)  # Shape: [batch_size, max_length]

        if hop_start_weights is not None:  # Train mode
            lgsf = torch.nn.LogSoftmax(
                dim=1
            )  # If there is no targeted span in the sentence, start_weights = end_weights = 0(vec)
            hop_start_loss = -torch.sum(
                hop_start_weights * lgsf(hop_start_logits), dim=-1
            )
            hop_end_loss = -torch.sum(hop_end_weights * lgsf(hop_end_logits), dim=-1)
            ans_start_loss = -torch.sum(
                ans_start_weights * lgsf(ans_start_logits), dim=-1
            )
            ans_end_loss = -torch.sum(ans_end_weights * lgsf(ans_end_logits), dim=-1)
            hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
            ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
```

可以看出，流程很简单，首先经过BERT前向运算，获得`semantics`，之后如果是训练模式则进行span predict的损失计算。

3.  评估模式的前向运算
```python
            K_hop, K_ans = 3, 1
            hop_preds = torch.zeros(
                batch_size, K_hop, 3, dtype=torch.long, device=device
            )  # (start, end, sen_num)
            ans_preds = torch.zeros(
                batch_size, K_ans, 3, dtype=torch.long, device=device
            )
            ans_start_gap = torch.zeros(batch_size, device=device)
            for u, (start_logits, end_logits, preds, K, allow) in enumerate(
                (
                    (
                        hop_start_logits,
                        hop_end_logits,
                        hop_preds,
                        K_hop,
                        allow_limit[0],
                    ),
                    (
                        ans_start_logits,
                        ans_end_logits,
                        ans_preds,
                        K_ans,
                        allow_limit[1],
                    ),
                )
            ):
                for i in range(batch_size):
                    if sep_positions[i, 0] > 0:
                        values, indices = start_logits[i, B_starts[i] :].topk(K)
                        for k, index in enumerate(indices):
                            if values[k] <= start_logits[i, 0] - allow:  # not golden
                                if u == 1: # For ans spans
                                    ans_start_gap[i] = start_logits[i, 0] - values[k]
                                break
                            start = index + B_starts[i]
                            # find ending
                            for j, ending in enumerate(sep_positions[i]):
                                if ending > start or ending <= 0:
                                    break
                            if ending <= start:
                                break
                            ending = min(ending, start + 10)
                            end = torch.argmax(end_logits[i, start:ending]) + start
                            preds[i, k, 0] = start
                            preds[i, k, 1] = end
                            preds[i, k, 2] = j
```

可以看出，每次预测hop span取的是 Top 3 ，ans span 取的是 Top 1（都需要大于阈值），而且抽取的位置都是在Sentence B,也就是论文中说到的`Para[x]`，从代码`values, indices = start_logits[i, B_starts[i] :].topk(K)`可以看出来。

# System 2
系统二的框架如下图所示：
![系统二结构图](https://img-blog.csdnimg.cn/20200502103136548.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hpYW5nSmlhb0p1bl8=,size_16,color_FFFFFF,t_70#pic_center)
系统二的主体就是GNN，主要是用来推理，在图中的所有节点中，推理出最终的答案节点。系统二的实现是`model.py`中的`CognitiveGNN`类

## 系统二输入与输出
*  `semantics`,`bundle.adj`，输入是一张图，semantics分别是图的隐层表示，以及图的邻接矩阵，GCN利用这些信息进行推理预测，不同问题类型的推理过程略有不同，比如答案为一个实体，那么使用的就是图中节点预测的交叉熵损失，如果答案为回答是否，那么就进行相似度计算，然后再计算交叉熵损失，代码如下：
```python
        if bundle.question_type == 0:  # Wh-
            pred = self.gcn(bundle.adj.to(device), semantics)
            ce = torch.nn.CrossEntropyLoss()
            final_loss = ce(
                pred.unsqueeze(0),
                torch.tensor([bundle.answer_id], dtype=torch.long, device=device),
            )
        else:
            x, y, ans = bundle.answer_id
            ans = torch.tensor(ans, dtype=torch.float, device=device)
            diff_sem = semantics[x] - semantics[y]
            classifier = self.both_net if bundle.question_type == 1 else self.select_net
            final_loss = 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                classifier(diff_sem).squeeze(-1), ans.to(device)
            )
```
## 系统二源码
整个系统二源码如下:
```python
class CognitiveGNN(nn.Module):
    def __init__(self, hidden_size):
        super(CognitiveGNN, self).__init__()
        self.gcn = GCN(hidden_size)
        self.both_net = MLP((hidden_size, hidden_size, 1))
        self.select_net = MLP((hidden_size, hidden_size, 1))

    def forward(self, bundle, model, device):
        batch = bundle_part_to_batch(bundle)
        batch = tuple(t.to(device) for t in batch)
        hop_loss, ans_loss, semantics = model(
            *batch
        )  # Shape of semantics: [num_para, hidden_size]
        num_additional_nodes = len(bundle.additional_nodes)

        if num_additional_nodes > 0:
            max_length_additional = max([len(x) for x in bundle.additional_nodes])
            ids = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            segment_ids = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            input_mask = torch.zeros(
                (num_additional_nodes, max_length_additional),
                dtype=torch.long,
                device=device,
            )
            for i in range(num_additional_nodes):
                length = len(bundle.additional_nodes[i])
                ids[i, :length] = torch.tensor(
                    bundle.additional_nodes[i], dtype=torch.long
                )
                input_mask[i, :length] = 1
            additional_semantics = model(ids, segment_ids, input_mask)

            semantics = torch.cat((semantics, additional_semantics), dim=0)

        assert semantics.size()[0] == bundle.adj.size()[0]

        if bundle.question_type == 0:  # Wh-
            pred = self.gcn(bundle.adj.to(device), semantics)
            ce = torch.nn.CrossEntropyLoss()
            final_loss = ce(
                pred.unsqueeze(0),
                torch.tensor([bundle.answer_id], dtype=torch.long, device=device),
            )
        else:
            x, y, ans = bundle.answer_id
            ans = torch.tensor(ans, dtype=torch.float, device=device)
            diff_sem = semantics[x] - semantics[y]
            classifier = self.both_net if bundle.question_type == 1 else self.select_net
            final_loss = 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
                classifier(diff_sem).squeeze(-1), ans.to(device)
            )
        return hop_loss, ans_loss, final_loss

```

整个前向运算过程就是：
* 根据输入的文本提取出`semantics`表达信息，作为抽出的答案节点和实体节点的表示
* 利用GCN进行推理，如果答案是预测节点，那么直接对每个节点预测其作为答案节点的概率即可，如果是是否回答，则先计算两个实体的`diff_sem`,然后再进行预测。

# 算法总体流程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200502110642545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hpYW5nSmlhb0p1bl8=,size_16,color_FFFFFF,t_70#pic_center)
这是论文中的CogQA的算法流程，每次都选择一个frontier节点，进行系统一的前向运算，得到next hop以及ans node，并加入到输出的图中，然后利用系统二对图的隐层表示进行更新，不断迭代，直到没有新的frontier节点或者图已经足够大，最后进行答案的预测即可。具体的代码流程在`cogqa.py`的`cognitive_graph_propagate`函数中，下面是其核心代码，可以看出`queue`就是frontier节点的队列，每次都是在队头出队一个frontier节点进行前向运算。
```python
    while len(queue) > 0:
        # visit all nodes in the frontier queue
        ids, segment_ids, input_mask, sep_positions, tokenized_alls, B_starts = construct_infer_batch(queue)
        hop_preds, ans_preds, semantics_preds, no_ans_logits = model1(ids, segment_ids, input_mask, sep_positions,
            None, None, None, None, 
            B_starts, allow_limit)  
        new_queue = []
        for i, x in enumerate(queue):
            semantics[x] = semantics_preds[i]
            # for hop spans
            for k in range(hop_preds.size()[1]):
                l, r, j = hop_preds[i, k]
                j = j.item()
                if l == 0:
                    break
                gold_ret.add((i2e[x], j)) # supporting facts
                orig_text = context[i2e[x]][j]
                pred_slice = tokenized_alls[i][l : r + 1]
                l, r = find_start_end_before_tokenized(orig_text, [pred_slice])[0]
                if l == r == 0:
                    continue    
                recovered_matched = orig_text[l: r]
                pool = context if setting == 'distractor' else (i2e[x], j)
                matched = fuzzy_retrieve(recovered_matched, pool, setting)    
                if matched is not None:
                    if setting == 'fullwiki' and matched not in e2i and n < 10 + max_new_nodes:
                        context_new = get_context_fullwiki(matched)
                        if len(context_new) > 0: # cannot resovle redirection
                            # create new nodes in the cognitive graph
                            context[matched] = context_new
                            prev.append([])
                            semantics.append(None)
                            e2i[matched] = n
                            i2e.append(matched)
                            n += 1
                    if matched in e2i and e2i[matched] != x:
                        y = e2i[matched]
                        if y not in new_queue and (i2e[x], j) not in prev[y]:
                            # new edge means new clues! update the successor as frontier nodes.
                            new_queue.append(y)
                            prev[y].append(((i2e[x], j)))
            # for ans spans
            for k in range(ans_preds.size()[1]):
                l, r, j = ans_preds[i, k]
                j = j.item()
                if l == 0:
                    break
                gold_ret.add((i2e[x], j))
                orig_text = context[i2e[x]][j]
                pred_slice = tokenized_alls[i][l : r + 1]
                l, r = find_start_end_before_tokenized(orig_text, [pred_slice])[0]
                if l == r == 0:
                    continue    
                recovered_matched = orig_text[l: r]
                matched = fuzzy_retrieve(recovered_matched, context, 'distractor', threshold=70)
                if matched is not None:
                    y = e2i[matched]
                    ans_nodes.add(y)
                    if (i2e[x], j) not in prev[y]:
                        prev[y].append(((i2e[x], j)))
                elif n < 10 + max_new_nodes:
                    context[recovered_matched] = []
                    e2i[recovered_matched] = n
                    i2e.append(recovered_matched)
                    new_queue.append(n)
                    ans_nodes.add(n)
                    prev.append([(i2e[x], j)])
                    semantics.append(None)
                    n += 1
        if len(new_queue) == 0 and len(ans_nodes) == 0 and allow_limit[1] < 0.1: # must find one answer
            # ``allow'' is an offset of negative threshold. 
            # If no ans span is valid, make the minimal gap between negative threshold and probability of ans spans -0.1, and try again.
            prob, pos_in_queue = torch.min(no_ans_logits, dim = 0)
            new_queue.append(queue[pos_in_queue])
            allow_limit[1] = prob.item() + 0.1
        queue = new_queue
```

# 训练数据的准备
最后想说一下训练数据的准备流程，源码中是`data..py`文件中的`convert_question_to_samples_bundle`，分别来看每个部分的作用

一、建立实体和索引的字典
```python
   context = dict(data['context']) # all the entities in 10 paragraphs
    gold_sentences_set = dict([((para, sen), edges) for para, sen, edges in data['supporting_facts']]) 
    e2i, i2e = {}, [] # entity2index, index2entity
    for entity, sens in context.items():
        e2i[entity] = len(i2e)
        i2e.append(entity)
    clues = [[]] * len(i2e) # pre-extracted clues
```
最开始的代码类似于词典的建立，其中`e2i、i2e`分别为实体到索引、索引到实体的映射字典。`gold_sentences_set `是`supporting_facts`的映射，之后会用来查询`clues[x]`

二、抽取出clues 线索集
```python
    # Extract clues for entities in the gold-only cogntive graph
    for entity_x, sen, edges in data['supporting_facts']:
        for entity_y, _, _, _ in edges:
            if entity_y not in e2i: # entity y must be the answer
                assert data['answer'] == entity_y
                e2i[entity_y] = len(i2e)
                i2e.append(entity_y)
                clues.append([])
            if entity_x != entity_y:
                y = e2i[entity_y]
                clues[y] = clues[y] + tokenizer.tokenize(context[entity_x][sen]) + ['[SEP]']
```

这部分代码也是建立一个字典，通过前面的`gold_sentences_set `，可以找出每个节点的clues，从而建立出`clues[x]`的线索字典

三、构建训练样本
```python
    # Construct training samples
    for entity, para in context.items():
        num_hop, num_ans = 0, 0
        tokenized_all = tokenized_question + clues[e2i[entity]]
        if len(tokenized_all) > 512: # BERT-base accepts at most 512 tokens
            tokenized_all = tokenized_all[:512]
            print('CLUES TOO LONG, id: {}'.format(data['_id']))
        # initialize a sample for ``entity''
        sep_position = [] 
        segment_id = [0] * len(tokenized_all)
        hop_start_weight = [0] * len(tokenized_all)
        hop_end_weight = [0] * len(tokenized_all)
        ans_start_weight = [0] * len(tokenized_all)
        ans_end_weight = [0] * len(tokenized_all)

        for sen_num, sen in enumerate(para):
            tokenized_sen = tokenizer.tokenize(sen) + ['[SEP]']
            if len(tokenized_all) + len(tokenized_sen) > 512 or sen_num > 15:
                break
            tokenized_all += tokenized_sen
            segment_id += [sen_num + 1] * len(tokenized_sen)
            sep_position.append(len(tokenized_all) - 1)
            hs_weight = [0] * len(tokenized_sen)
            he_weight = [0] * len(tokenized_sen)
            as_weight = [0] * len(tokenized_sen)
            ae_weight = [0] * len(tokenized_sen)
            if (entity, sen_num) in gold_sentences_set:
                edges = gold_sentences_set[(entity, sen_num)]
                intervals = find_start_end_after_tokenized(tokenizer, tokenized_sen,
                    [matched  for _, matched, _, _ in edges])
                for j, (l, r) in enumerate(intervals):
                    if edges[j][0] == answer_entity or question_type > 0: # successive node edges[j][0] is answer node
                        as_weight[l] = ae_weight[r] = 1
                        num_ans += 1
                    else: # edges[j][0] is next-hop node
                        hs_weight[l] = he_weight[r] = 1
                        num_hop += 1
            hop_start_weight += hs_weight
            hop_end_weight += he_weight
            ans_start_weight += as_weight
            ans_end_weight += ae_weight
            
        assert len(tokenized_all) <= 512
        # if entity is a negative node, train negative threshold at [CLS] 
        if 1 not in hop_start_weight:
            hop_start_weight[0] = 0.1
        if 1 not in ans_start_weight:
            ans_start_weight[0] = 0.1

        ids.append(tokenizer.convert_tokens_to_ids(tokenized_all))
        sep_positions.append(sep_position)
        segment_ids.append(segment_id)
        hop_start_weights.append(hop_start_weight)
        hop_end_weights.append(hop_end_weight)
        ans_start_weights.append(ans_start_weight)
        ans_end_weights.append(ans_end_weight)
```

这部分代码作用是从训练数据中的`context`中构建训练样本，`context`有十个实体，每个实体都有一段文字，所以每条数据为**当前问题 + clue[x] + para[x]**，一个batch则为这**十条实体组成的数据**

四、添加负采样节点
```python
    # Construct negative answer nodes for task #2(answer node prediction)
    n = len(context)
    edges_in_bundle = []
    if question_type == 0:
        # find all edges and prepare forbidden set(containing answer) for negative sampling
        forbidden = set([])
        for para, sen, edges in data['supporting_facts']:
            for x, matched, l, r in edges:
                edges_in_bundle.append((e2i[para], e2i[x]))
                if x == answer_entity:
                    forbidden.add((para, sen))
        if answer_entity not in context and answer_entity in e2i:
            n += 1
            tokenized_all = tokenized_question + clues[e2i[answer_entity]]
            if len(tokenized_all) > 512:
                tokenized_all = tokenized_all[:512]
                print('ANSWER TOO LONG! id: {}'.format(data['_id']))
            additional_nodes.append(tokenizer.convert_tokens_to_ids(tokenized_all))

        for i in range(neg):
            # build negative answer node n+i
            father_para = random.choice(list(context.keys()))
            father_sen = random.randrange(len(context[father_para]))
            if (father_para, father_sen) in forbidden:
                father_para = random.choice(list(context.keys()))
                father_sen = random.randrange(len(context[father_para]))
            if (father_para, father_sen) in forbidden:
                neg -= 1
                continue
            tokenized_all = tokenized_question + tokenizer.tokenize(context[father_para][father_sen]) + ['[SEP]']
            if len(tokenized_all) > 512:
                tokenized_all = tokenized_all[:512]
                print('NEG TOO LONG! id: {}'.format(data['_id']))
            additional_nodes.append(tokenizer.convert_tokens_to_ids(tokenized_all))
            edges_in_bundle.append((e2i[father_para], n))
            n += 1
```

对于系统二，我们需要预测答案节点，所以可以添加负节点来提高训练效果

五、计算实体节点和答案节点邻接矩阵
```python
    if question_type >= 1:
        for para, sen, edges in data['supporting_facts']:
            for x, matched, l, r in edges:
                if e2i[para] < n and  e2i[x] < n:
                    edges_in_bundle.append((e2i[para], e2i[x]))
                    
    assert n == len(additional_nodes) + len(context)
    adj = torch.eye(n) * 2
    for x, y in edges_in_bundle:
        adj[x, y] = 1
    adj /= torch.sum(adj, dim=0, keepdim=True)
```

邻接矩阵会在GCN进行推理的时候用到，根据推理关系，可以在不同节点中建立边，得到邻接矩阵，并对其归一化，得到归一化的邻接矩阵

# 后记
&emsp;&emsp;说实话读完这篇论文，感觉接受了一次洗礼，原来MRC还能这样做，不得不说将multi-hop和图推理结合起来真的让人眼前一亮，这种方式可以解决前面说的原始的基于span extraction的方法难以进行复杂推理的缺点，我个人想法，关于图在文本理解方面可能还有几个方向：
1. 图关系的补全，从当前文本可以构建一个图，之后利用外部知识库对图进行补全，完善整个图的结构
2. 图的匹配，可以对不同答案构成的不同图进行匹配，找到最合适的那个

总之，这篇论文信息量很大，上面的都是我自己的理解分享，源码也是贴了很小一部分，难免有错，如果小伙伴们感兴趣的话，希望在评论区留言呀~

