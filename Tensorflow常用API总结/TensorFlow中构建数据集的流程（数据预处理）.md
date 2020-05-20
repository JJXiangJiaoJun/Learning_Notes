[TOC]
# 前言
* 参照BERT源码中`create_pretraining_data.py`中从原始文本文件读取数据，写入到`TFRecord`文件中，并从`TFRecord`文件中读取数据并输入到网络中运算的整个过程。

# 整体流程
1. 从原始文本文件中读取数据（字符串）
2. 将原始字符串数据进行编码，比如id映射，截断，mask等等预处理，得到我们输入网络中的数据格式
3. 将上一步中得到的经过预处理后的数据，通过`tf.io.TFRecordWriter`写入到`TFRecord`文件中
4. 编写`input_fn`，从`TFRecord`文件中读取数据输入到模型中


# 各个处理步骤用到的函数和类

## 一、从原始文件读取数据
* 这一步的主要作用就是从原始文件中读取数据，转化为一条条的`InputExample`（这时候还是文本数据）

### `InputExample`类
* `InputExample`类用来保存原始数据，每一个`InputExample`实例保存一条数据，**我们整个数据集就是一个`InputExample`的列表**,这个将会**作为后续数据预处理的输入**

```python
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
```

### `DataProcessor`类
* `DataProcessor`类用来读取原始数据并生成`InputExample`，类似于Pytorch中的`Dataset`。通过这个类，我们可以生成我们数据集也就是`InputExample`列表

```
class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        if labels is not None:
            try:
                # 支持从文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                self.labels = set(self.labels) # to set
            except Exception as e:
                print(e)
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines
```

* `_read_data`方法：在这个方法实现原始数据的读取与处理
* `_create_example`方法：这个方法创建`InputExample`列表，作为数据集

## 二、对处理好的数据进行预处理

### `InputFeatures` 类
* `InputFeatures`类的用来作为`TFRecordWriter`的Features，每一个`InputFeatures`保存的是经过预处理之后的数据，里面的数据形式就是输入模型中的数据形式

```python
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask
```

### `convert_single_example`函数
*  将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到`InputFeatures`对象中,这个函数的返回值为`InputFeatures`对象

```python
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)
```

### `filed_based_convert_examples_to_features`函数

* 其中会调用`convert_single_example`函数，对每条数据转化为输入网络的`InputFeatures`形式，然后使用`tf.python_io.TFRecordWriter`写入到`TFRecord`文件中

```python
def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
```

## 三、编写从`TFRecord`文件中读取数据的输入函数

### `file_based_input_fn_builder`
* 这个函数返回一个`tf.data.TFRecordDataset`对象

```python
def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn

```

# `create_pretraining_data.py`中函数分析

## `TrainingInstance`类
* 这个类对应于`InputFeature`类，用来结构化输入模型的数据
```python

```

## `create_training_instances`函数
* 这个函数对应于`DataProcess`类，是从原始数据中读取并解析出数据集
```
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng)
```

## `create_instances_from_document`函数
* 这个函数对应于`preprocess`函数，对数据集进行预处理，并结构化为`TrainingInstance`类
```python
def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
```
## `write_instance_to_example_files`函数
* 这个函数把`TrainingInstance`写入文件中

```python
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files)
```