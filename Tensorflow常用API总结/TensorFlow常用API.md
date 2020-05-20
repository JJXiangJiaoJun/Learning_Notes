[TOC]
# `tf.flags`

## 作用
使用命令行来指定我们代码中的参数，例如`python train.py --batch_size 12 ...`，类似python中的argparse库，使用flags定义命令行的参数(参数统一保存在`flags.FLAGS`中)

## 参数类型
```python
tf.flags.DEFINE_integer(
    name, default, help, lower_bound=None,
    upper_bound=None,flag_values=_flagvalues.FLAGS, **args
)
```
使用tf.flags.DEFINE_integer来定义一个integer类型参数，还有许多常用的类型，例如:
* DEFINE_float
* DEFINE_integer
* DEFINE_string
* DEFINE_enum
* DEFINE_bool
* ....

[API文档](https://tensorflow.google.cn/api_docs/python/tf/compat/v1/flags?hl=en)

## 使用方法
* 调用函数创建一个flags对象
```python
flags = tf.app.flags
#（或者tf.flags）
```
* 定义需要的参数
```python
tf.flags.DEFINE_string("label_file", None, "File containing output labels")
```
* 获取`flags.FLAGS`对象，所有定义的参数都保存在`flags.FLAGS`中

```python
FLAGS = flags.FLAGS
```
* 运行tf.app.run()

```python
tf.app.run()
```

# `tf.data.Dataset`
## 作用
`tf.data.Dataset`用来读取数据，Dataset可以看作是相同类型“元素”的有序列表。使用`tf.data.Dataset.from_tensor_slices`来创建一个dataset

```python
import tensorflow as tf
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                       
        "b": np.random.uniform(size=(5, 2))
    }
)
```

## 支持操作
* Map map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset，如我们可以对dataset中每个元素的值加1:
```python
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0
```

* batch batch就是将多个元素组合成batch，如下面的程序将dataset中的每个元素组成了大小为32的batch:
```python
dataset = dataset.batch(32)
```
* shuffle shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小：
```python
dataset = dataset.shuffle(buffer_size=10000)
```
* repeat repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch，假设原先的数据是一个epoch，使用repeat(5)就可以将之变成5个epoch：
```python
dataset = dataset.repeat(5)
```
* apply apply接受一个Dataset对象，并且会返回一个经过了`transformation_func`的Dataset
```python
dataset = (dataset.map(lambda x: x ** 2)
           .apply(group_by_window(key_func, reduce_func, window_size))
           .map(lambda x: x ** 3))
```

## Dataset其他创建方法
* `tf.data.TextLineDataset()：` 这个函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素就对应了文件中的一行。可以使用这个函数来读入CSV文件。
* `tf.data.FixedLengthRecordDataset()：`这个函数的输入是一个文件的列表和一个record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。通常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式
* `tf.data.TFRecordDataset()：`顾名思义，这个函数是用来读TFRecord文件的，dataset中的每一个元素就是一个TFExample。


# `tf.train.list_variables`
## 作用
返回所有在`checkpoint`中变量的列表
## 使用方法
```
tf.train.list_variables(ckpt_dir_or_file)
```

* 返回值 变量 `(name,shape)`的列表

# `tf.trainable_variables()`
## 作用
* 获取所有创建时`trainable = True`的变量

## 使用方法
```
tf.trainable_variables(scope=None)
```
* 返回值，A list of Variable objects.
* 类似的函数，`tf.all_variables(), tf.global_variables()`

# `tf.train.init_from_checkpoint`
## 作用
* 用来从`checkpoint`初始化保存的参数

## 用法
```python
tf.train.init_from_checkpoint(
    ckpt_dir_or_file,
    assignment_map
)
```

```python
# Say, '/tmp/model.ckpt' has the following tensors:
#  -- name='old_scope_1/var1', shape=[20, 2]
#  -- name='old_scope_1/var2', shape=[50, 4]
#  -- name='old_scope_2/var3', shape=[100, 100]

# Create new model's variables
with tf.compat.v1.variable_scope('new_scope_1'):
  var1 = tf.compat.v1.get_variable('var1', shape=[20, 2],
                         initializer=tf.compat.v1.zeros_initializer())
with tf.compat.v1.variable_scope('new_scope_2'):
  var2 = tf.compat.v1.get_variable('var2', shape=[50, 4],
                         initializer=tf.compat.v1.zeros_initializer())
  # Partition into 5 variables along the first axis.
  var3 = tf.compat.v1.get_variable(name='var3', shape=[100, 100],
                         initializer=tf.compat.v1.zeros_initializer(),
                         partitioner=lambda shape, dtype: [5, 1])

# Initialize all variables in `new_scope_1` from `old_scope_1`.
init_from_checkpoint('/tmp/model.ckpt', {'old_scope_1/': 'new_scope_1'})

# Use names to specify which variables to initialize from checkpoint.
init_from_checkpoint('/tmp/model.ckpt',
                     {'old_scope_1/var1': 'new_scope_1/var1',
                      'old_scope_1/var2': 'new_scope_2/var2'})

# Or use tf.Variable objects to identify what to initialize.
init_from_checkpoint('/tmp/model.ckpt',
                     {'old_scope_1/var1': var1,
                      'old_scope_1/var2': var2})

# Initialize partitioned variables using variable's name
init_from_checkpoint('/tmp/model.ckpt',
                     {'old_scope_2/var3': 'new_scope_2/var3'})

# Or specify the list of tf.Variable objects.
init_from_checkpoint('/tmp/model.ckpt',
                     {'old_scope_2/var3': var3._get_variable_list()})
```
### 参数
* `ckpt_dir_or_file`：checkpoint的目录或者路径
* `assignment_map`：字典类型

## 文档
* [tf.train.init_from_checkpoint](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/train/init_from_checkpoint.md)


# `tf.contrib.data.map_and_batch`
## 作用
融合了 `map和batch`操作
## 文档
* [tf.contrib.data.map_and_batch](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/contrib/data/map_and_batch.md)

# `tf.contrib.data.parallel_interleave`
## 文档
* [tf.contrib.data.parallel_interleave](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/contrib/data/parallel_interleave.md)

# `tf.control_dependencies`
## 作用
* 有些程序中我们想要指定某些操作执行的依赖关系，这时我们可以使用`tf.control_dependencies()`实现

## 使用方法
```python
 with tf.control_dependencies([a, b, c]):
      # `d` and `e` will only run after `a`, `b`, and `c` have executed.
      d = ...
      e = ...

```

* 这样的话只有在`a、b、c`被执行之后，才会运算`d,e`
# `tf.train.get_or_create_global_step()`

## 作用
* 这个函数主要用于返回或者创建（如果有必要的话）一个全局步数的tensor。参数只有一个，就是图，如果没有指定那么就是默认的图。

## 使用方法
```python
global_steps = tf.train.get_or_create_global_step()
```

# `tf.assgin()`
## 作用
* 赋值操作，为TensorFlow中的变量做赋值操作

## 使用方法
```python
tf.assign(ref, value, 
    validate_shape=None, 
    use_locking=None, name=None)
```

# `tf.assign_add`
## 作用
* 增加一个Tensorflow中的变量的值

## 使用方法
```python
tf.assign_add(ref,value,
        use_locking=None,name=None)
```

* 比如说,`global_step`就是在`optimizer.apply_gradient()`中，使用`tf.assign_add(global_step,1)`来实现每个训练流程之后自动+1操作。

# `tf.eye`
## 作用
* 创建一个对角矩阵

## 使用方法
```python
tf.eye(
    num_rows,
    num_columns=None,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)

# Construct one identity matrix.
tf.eye(2)
==> [[1., 0.],
     [0., 1.]]

# Construct a batch of 3 identity matricies, each 2 x 2.
# batch_identity[i, :, :] is a 2 x 2 identity matrix, i = 0, 1, 2.
batch_identity = tf.eye(2, batch_shape=[3])

# Construct one 2 x 3 "identity" matrix
tf.eye(2, num_columns=3)
==> [[ 1.,  0.,  0.],
     [ 0.,  1.,  0.]]

```
## 官方文档
* [tf.eye](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/eye.md)

# `tf.nn.l2_normalize`

## 作用
* 对向量指定的维度进行`normalization`操作

## 使用方法
```python
tf.math.l2_normalize(
    x,
    axis=None,
    epsilon=1e-12,
    name=None,
    dim=None
)
```

## 官方文档
* [tf.nn.l2_normalize](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/math/l2_normalize.md)


# `tf.nn.log_softmax`
## 作用
* 实现`log_softmax`的激活函数操作

## 用法
```python
tf.nn.log_softmax(
    logits,
    axis=None,
    name=None,
    dim=None
)
```
# 数据IO
* 数据io操作API在`tf.io`下面


# 常用损失函数
## `tf.nn.sigmoid_cross_entropy_with_logits`
### 作用
* 计算给定`logits`下的`sigmoid_cross_entropy`

### 使用方法
```python
tf.nn.sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

* 假设 `x = logits` ，`z = labels`，那么计算方法如下:
```python
  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + log(1 + exp(-x))
= x - x * z + log(1 + exp(-x))
```

* 注意点：`labels` 和 `logits` Tensor形状要一样

## `tf.nn.softmax_cross_entropy_with_logits`
### 作用
* 计算给定`prob`下`softmax cross_entropy`

### 使用方法
```python
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None,
    axis=None
)
```

* **注意这里`labels`要为`one_hot`形式**

## `tf.nn.sparse_softmax_cross_entropy_with_logits`

### 作用
* 计算给定`logits`下的`sparse cross entropy`

### 使用方法
```python
tf.nn.sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

* **注意这里的`labels`**就不是`one-hot`形式了，而是在`[0,num_classes)`表示

# 常用的获取某些特殊变量的API

## `tf.GraphKeys`

### 作用
* 使用`graph collections`时的标准名字

### 用法

有下列的`standard keys`：
* `GLOBAL_VARIABLES` ：`tf.GraphKeys.GLOBAL_VARIABLES`
* `LOCAL_VARIABLES`
* `MODEL_VARIABLES`
* `TRAINABLE_VARIABLES`
* `SUMMARIES`
* `QUEUE_RUNNERS`
* `MOVING_AVERAGE_VARIABLES`
* `REGULARIZATION_LOSSES`

## `tf.get_collection`

### 作用
* 获取某一个`collections`

### 使用方法
```python
tf.get_collection(
    key,
    scope=None
)
```

* 返回值
    * 一个Tensor的列表   

# Tensor运算API

## `tf.clip_by_global_norm`

### 作用
* 对Tensor列表根据其范数之和进行裁剪
* 可以用于梯度裁剪

### 使用方法
```python
tf.clip_by_global_norm(
    t_list,
    clip_norm,
    use_norm=None,
    name=None
)
```

* 其内部计算方法如下
    * `t_list[i] * clip_norm / max(global_norm, clip_norm)`
    * `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`
* 函数返回值
    * **`list_clipped`**: 进行裁剪过后的Tensor列表
    * **`global_norm`** ：一个标量，表示`global norm`

## `tf.convert_to_tensor`
### 作用
* 将给定的`value`转化为`Tensor`对象

### 使用方法
```python
tf.convert_to_tensor(
    value,
    dtype=None,
    name=None,
    preferred_dtype=None,
    dtype_hint=None
)
```

```python
import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```

## `tf.group`
### 作用
* 创建一个包含多个`operations`的`op`

### 用法
```python
tf.group(
    *inputs,
    **kwargs
)
```

* 当这个`op`完成时，所有在`inputs`中的`op`也已经完成

## `tf.shape`
### 作用
* 返回一个`shape`的`Tensor`

### 使用方法
```python
tf.shape(
    input,
    name=None,
    out_type=tf.dtypes.int32
)

t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t)  # [2, 2, 3]
```