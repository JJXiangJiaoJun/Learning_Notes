import tensorflow as tf
import math
import os

def create_initializer(initializer_range = 0.02):
    return tf.truncated_normal_initializer(stddev = initializer_range)


def embedding_lookup(input_ids,vocab_size,embedding_size,initializer,scope,
                    reuse=None,dtype = tf.float32):
    with tf.variable_scope(scope,reuse = reuse):
        lookup_table = tf.get_variables('lookup_table',[vocab_size,embedding_size],
                                        initializer = initializer,dtype=dtype)
    return tf.nn.embedding_lookup(lookup_table,input_ids)


def positional_embedding(pos_seq,inv_freq):
    pos_seq = tf.expand_dims(pos_seq,0)
    inv_freq = tf.expand_dims(inv_freq,1)
    sin_cos_seq = pos_seq*inv_seq
    output_embedding = tf.concat([tf.sin(sin_cos_seq),tf.cos(sin_cos_seq)],-1)
   
    return output_embedding

def positional_encoding(max_length,embedding_size):
    freq_seq = tf.range(0,embedding_size,2.0)
    inv_freq = 1.0/(10000**(freq_seq/embedding_size))

    pos_seq = tf.range(0,max_length)
    return positional_embedding(pos_seq,inv_freq)

def attention_layer(input_tensor,
                    num_attention_head,
                    attention_head_size):
    """
    input_tensor : [Batch_size , seq_length , width] [B,N,W]
    num_attention_head : K
    attention_head_size : H

    """
    input_shape = input_tensor.shape.as_list()
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    def transpose_for_score(input_tensor,batch_size,seq_length,
                            num_attention_head,attention_head_size):
        input_tensor = tf.reshape(input_tensor,[batch_size,seq_length,
                            num_attention_head,attention_head_size])
        output_tensor = tf.transpose(input_tensor,[0,2,1,3])
        return output_tensor

    #[B*N,W]
    input_tensor = tf.reshape(input_tensor,[-1,width])

    #[B*N,K*H]
    query = tf.layer.dense(
        input_tensor,
        num_attention_head*attention_heads_size,
        name = 'query'
        )

    key = tf.layer.dense(
        input_tensor,
        num_attention_head*attention_heads_size,
        name = 'key'
        )
    value = tf.layer.dense(
    input_tensor,
    num_attention_head*attention_heads_size,
    name = 'value'
    )
    #[B,K,N,H]
    query =  transpose_for_score(query,batch_size,seq_length,
                            num_attention_head,attention_head_size)
    key =  transpose_for_score(key,batch_size,seq_length,
                            num_attention_head,attention_head_size)
    #[B,K,N,N]
    attention_scores = tf.matmul(query,key,transpose_b =True)
    attention_scores = tf.multiply(attention_scores,1.0/math.sqrt(attention_head_size))
    attention_probs = tf.nn.softmax(attention_scores,axis = -1)

    value = transpose_for_score(value,batch_size,seq_length,
                            num_attention_head,attention_head_size)

    context_output =  tf.matmul(attention_probs,value)

    return context_output
