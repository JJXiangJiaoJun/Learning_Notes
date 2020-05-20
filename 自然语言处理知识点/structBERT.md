[TOC]
# Word Structural Objective
* inner-sentence，原本的Mask词加上打乱subsequence的单词顺序，并且希望模型还原。（论文中K=3,5%）
* 思想是希望Language Model能够自动纠正错误的词语顺序
* 主要针对的是单个句子的任务


# Sentence structural Objective
* inter-sentence.主要针对的是sentence-pair的任务，将原本的NSP任务，改成一个三分类任务。
* 1/3的概率：`（S_1,S_2）`是上下句，分类标签为1
* 1/3的概率，`(S_1,S_2)`是上下句反序，分类标签为2
* 1/3的概率，`(S_1,S_rand)`是不同文档的句子。