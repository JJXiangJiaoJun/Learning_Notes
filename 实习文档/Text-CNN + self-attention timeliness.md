# Text-CNN + self-attention timeliness

# 1. BERT模型实验

代码地址 : http://lego.sm.cn/ide?project_id=3104

## 1.1. Finetune任务

&emsp;&emsp;使用2kw标注文本数据进行训练，标注方式为下列形式:
* 0 : 表示当前query没有时效性需求
* 1 : 表示当前query有时效性需求

原始数据格式如下,Finetune任务为**2分类问题**:

```
[dat]
input_words=0:101,3736,2123,3749,6756,2399,3466,102
segment_ids=0:0,0,0,0,0,0,0,0
label=0:1
[dat]
input_words=0:101,4125,2512,722,7885,782,150,5286,2797,1215,1062,2147,1059,2506,102
segment_ids=0:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
label=0:0
```



### 1.1.1. 训练数据
&emsp;&emsp;训练数据通过lgb时效性小模型对线上query进行打分，根据分值对数据进行标注，分值-3至-2之间为负例，大于-1为正例，从中随机抽取2千万作为训练数据，正负比例1:1.
&emsp;&emsp;数据地址: hdfs://sm-hadoop/user/kangping.yinkp/zhuweida/bert/data/train.2kw

### 1.1.2. 训练参数
|参数名称|参数值|
|----|---|
|train_batch_size|64|
|learning rate|0.000005|
|optimizer|ADAM_OPTIMIZER|
|num_epochs|2|

### 1.1.3. 模型参数
&emsp;&emsp;训练采用BERT 8 hidden layer 模型,加载Google中文预训练模型 **（字粒度）** ，具体参数配置如下:

```json
{
  "attention_probs_dropout_prob": 0.1, 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 8, 
  "type_vocab_size": 2, 
  "vocab_size": 21128
}
```

### 1.1.4. 模型地址
&emsp;&emsp;地址 : hdfs://sm-hadoop/tmp/zqh/bert/train/checkpoint_adam/

----

## 1.2. 测试

### 1.2.1. 测试数据
&emsp;&emsp;人工标注的三千条数据，正负比例2:1，**该数据存在部分数据标注错误** (未进行人工review)<br/>
&emsp;&emsp;地址 : hdfs://sm-hadoop/tmp/zhuweida/bert/data/test.3k.csv

### 1.2.2. 测试结果

**mAP = 0.936388**

|分数阈值|Precision|Recall|f1score|
|------|-------|-----|-----|
|0.7|0.813127|**0.944915**|**0.874081**|
|0.8|0.824718|0.929555|0.874004|
|0.9|0.844422|0.894068|0.868536|
|0.95|**0.872886**|0.847458|0.859984|
### 1.2.3. PR曲线
![BERT_PR_Curve.jpg]) 

------


# 2. textCNN-self_attention 蒸馏实验

代码地址 : http://lego.sm.cn/ide?project_id=3149

## 2.1. 蒸馏训练

&emsp;&emsp;使用由BERT_8L预测得到的8 million 数据，对textCNN-self_attention **(textCNN + self_attention)** 进行蒸馏实验,训练输入采用分词（BERT中为单字）,具体数据格式如下,其中scores为BERT的预测结果,用于计算textCNN的soft loss:

```
[dat]
input_word=53:101,3736,2123,3749,6756,2399,3466,102,0,0,0,0,0,0,0,0
label=1:1
score=28:-2.28962254524,3.14288878441
[dat]
input_word=71:101,4125,2512,722,7885,782,150,5286,2797,1215,1062,2147,1059,2506,102,0
label=1:0
score=28:3.17638039589,-4.10340070724
```

### 2.1.1. 训练数据
&emsp;&emsp;数据地址: hdfs://sm-hadoop/user/kangping.yinkp/zengqinghong/timeliness_bert_cnn/textCNN_splitword

### 2.1.2. 训练参数

&emsp;&emsp;训练时使用GPU分布式训练
|参数名称|参数值|
|----|---|
|train_batch_size|8192|
|learning rate|0.0001|
|optimizer|ADAM_OPTIMIZER|
|num_epochs|4|

### 2.1.3. 模型参数
&emsp;&emsp;训练采用Text-CNN + self attention结构 **（词粒度）** ，具体参数配置如下:

```json
{
  "sequence_len": 10, 
  "filter_sizes": [3,4,5], 
  "num_filters": 256, 
  "hidden_size": 128, 
  "distill_temperature":5.0,
  "dropout_rate": 0.3, 
  "hard_weight":0.5,
  "embedding_size": 128, 
  "vocab_size": 948147
}
```

### 2.1.4. 模型地址
&emsp;&emsp;地址 : hdfs://sm-hadoop/tmp/zqh/textCNN/train/splitword_attention/

----

## 2.2. 测试

### 2.2.1. 测试数据
&emsp;&emsp;人工标注的三千条数据，正负比例2:1，**该测试数据已经经过人工review**<br/>
&emsp;&emsp;地址 : hdfs://sm-hadoop/user/kangping.yinkp/zengqinghong/timeliness_bert_cnn/eval_3k/

### 2.2.2. 测试结果

**mAP = 0.956905**


|分数阈值|Precision|Recall|f1score|
|------|-------|-----|-----|
|0.7|0.896897|**0.909927**|**0.903365**|
|0.8|0.906640|0.879419|0.892822|
|0.9|**0.927433**|0.816949|0.868692|


### 2.2.3. PR曲线
![textCNN_PR_Curve.jpg]() 

------
