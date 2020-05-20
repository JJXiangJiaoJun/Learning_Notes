```python
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    sequence_length =
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 定义模型数据输出结构 定长的sequence_length
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                # 每一个词都是embedding_size长度的特征向量 (18758,128)
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            #根据词的下标，获取它们的word2vec。
            #embedded_chars的shape[sequence_length, embedding_size]
            # (none,56,128) sequence_length = 56
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            #扩充维度 相当于一个1维的通道数
            # [None,56,128,1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # filter_size 分别为3 4 5
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d( # [None,56-3+1,1,128] [None,56-4+1,1,128] [None,56-5+1,1,128]
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool( #[None,1,1,128]
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1], #[1,54,1,1] [1,53,1,1] [1,52,1,1]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print(pooled)
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 全连接dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



```

# TextCNN缺点
* Textcnn最大的问题全局max pooling丢失了结构信息，很难发现文本中的转折关系等复杂模式。
* textcnn只知道关键词是否在文本中出现了，以及相似度强度分布，不可能知道关键词出现了几次，以及出现这些关键词的顺序。

# DPCNN
* 这里作者将TextCNN的包含多尺寸卷积滤波器的卷积层的卷积结果称之为Region embedding
* 可以学习到更长距离的依赖
* DPCNN中固定featrue map数量，认为句子的有些词语可以进行合并。