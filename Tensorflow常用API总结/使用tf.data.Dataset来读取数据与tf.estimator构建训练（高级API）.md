[TOC]
# 说明
* 比如在BERT源码中，需要从`TFRecord`文件中读取训练数据并输入到网络中，`tf.data.Dataset`属于Tensorflow 高级API，一般与`tf.estimator`搭配使用
* 即目前的标准用法为 `tf.data.Dataset + tf.estimator.Estimator`


# tf.estimator.Estimator
## 官方文档地址
* [tf.estimator.Estimator](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/estimator/Estimator.md)

## 作用

* 可以实例化一个`Estimator `类，用来`train`或者`eval`Tensorflow模型,比如下面用法


```
estimator = tf.estimator.DNNClassifier(
    feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
    hidden_units=[1024, 512, 256],
    warm_start_from="/path/to/checkpoint/dir")
    
estimator.train(......)
```

## 函数原型介绍

### `__init__`

```python
__init__(
    model_fn,
    model_dir=None,
    config=None,
    params=None,
    warm_start_from=None
)
```
* 构造函数如上，**重要的是`model_fn`**

#### 参数说明
* **`model_fn`**:我们应该定义一个`model_fn`函数，返回`tf.estimator.EstimatorSpec`对象，`model_fn`参数如下
    * `features`：这是从`input_fn`中返回的第一个item，应该为一个`tf.Tensor`或者`dict`
    * `labels`：这是从`input_fn`中返回的第二个item，同上，有可能为`None`
    * `mode`，可选值。表示当前是什么模式
    * `params`,`dict`对象，超参数
    * `config`，`estimator.RunConfig `对象

## 注意点
* 从上面可以看出，我们要用`tf.estimator.Estimator`，需要定义两个函数：
    * `model_fn`，对应`tf.estimator.EstimatorSpec`对象
    * `input_fn`，对应`tf.data.Dataset`对象


# 使用tf.data.Dataset与tf.estimator等高级API构建训练过程

## 一、定义`input_fn`（数据输入部分）
* 这个函数需要返回一个`tf.data.Dataset`对象，给之后`tf.estimator.Estimator`对象`train,predict`等方法使用
* 例如BERT中定义的`input_fn`代码如下：

```python
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""


  # 这是TFRecord文件中保存数据的形式
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  } 
  
  # 这个函数用来解析TFRecord文件中读取到的数据
  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example
  
  # 这个函数返回一个tf.data.Dataset对象
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    
    # apply对于每个record都会调用，进行解码
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn
```

* `input_fn`输入参数可以有下列三个，会从`estimator`实例化时传递过来
    * `params`，
    * `mode`
    * `config`

## 二、定义`model_fn`函数（模型定义部分，其中包含loss计算，train_op实现等）
* 这个函数需要**返回一个`tf.estimator.EstimatorSpec`对象**，这个函数中需要完成下面功能：
    * 获得输入数据并进行模型的前向运算得到`loss`
    * 定义优化器获得`trian_op`
    * 必要的话在其中进行参数的加载，初始化
    * 对于不同的`mode`实现不同的流程
* BERT中定义的`model_fn`如下:

```python
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn
```

## 三、构建`tf.estimator.Estimator`对象，并进行训练等操作
* 通过上面两步定义了`input_fn`、`model_fn`就可以实例化`tf.estimator.Estimator`对象了

```
# 创建model_fn
model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

# 实例化estimator对象
estimator = tf.contrib.tpu.TPUEstimator(
  use_tpu=FLAGS.use_tpu,
  model_fn=model_fn,
  config=run_config,
  train_batch_size=FLAGS.train_batch_size,
  eval_batch_size=FLAGS.eval_batch_size,
  predict_batch_size=FLAGS.predict_batch_size)
  
# 创建input_fn
train_input_fn = file_based_input_fn_builder(
    input_file=train_file,
    seq_length=FLAGS.max_seq_length,
    is_training=True,
    drop_remainder=True)

# 使用estimator进行训练
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
```
