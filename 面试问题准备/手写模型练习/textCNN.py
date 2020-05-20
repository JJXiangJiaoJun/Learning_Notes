import tensorflow as tf
import numpy as np

def TextCNN(input_ids,labels,vocab_size,embedding_size,
            filter_sizes,num_filters,drop_rate,num_classes):
    
    seq_length = input_ids.shape.as_list()[1]
    with tf.variable_scope('lookup_table'):
        lookup_table = tf.get_variable('embedding',[vocab_size,embedding_size],
                                dtype = tf.float32,initializer = tf.random_normal_initializer())
        embedding_output = tf.nn.embedding_lookup(lookup_table,input_ids)
        # [B,N,H] -> [B,N,H,1]
        embedding_output = tf.expand_dims(embedding_output,-1)
    
    # self attention

    pooled_output = list()

    for i,filter_size in enumerate(filter_sizes):
        with tf.variable_scope('conv-maxpool-%d'%i):
            filter_shape = [fliter_size,embedding_size,1,num_filters]
            W = tf.get_variable('conv-%d'%filter_size,filter_shape,initializer = tf.random_normal_initializer())
            b = tf.get_variable('bias',shape = [num_filters],initializer = tf.constant(0.0))

            conv = tf.nn.conv2d(
                embedding_output,
                W,
                strides = [1,1,1,1],
                padding = 'VALID',
                name = 'conv2d'
            )

            h = tf.nn.relu(tf.nn.bias_add(conv,b))
            pool_out = tf.nn.max_pool2d(
                h,
                ksize = [1,seq_length - filter_size + 1,1,1],
                padding = 'VALID',
                name = 'maxpool'
            )
            pooled_output.append(pool_out)
    
    #[B,1,1,N]
    concat_output = tf.concat(pooled_output,axis = -1)
    h_pool = tf.reshape(concat_output,[-1,num_filters])
    with tf.variable_scope('dropout'):
        h_drop = tf.layers.dropout(h_pool,drop_rate)
    with tf.variable_scope('dense'):
        #[B,num_class]
        output = tf.layers.dense(
            h_drop,
            num_classes,

        )
    
    loss = tf.nn.softmax_cross_entropy_with_logits(output,labels)


    