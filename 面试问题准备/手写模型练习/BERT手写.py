import tensorflow as tf
import math

vocab_size = 100000
embedding_size = 768

def create_initializer(initializer_range = 0.02):
    return tf.truncated_normal_initializer(stddev = initializer_range)

# 词嵌入

def embedding_lookup(input_ids,vocab_size,embedding_size,initializer,
                    scope = 'embedding',reuse=None,dtype = tf.float32):
    with tf.variable_scope(scope,reuse = reuse):
        lookup_table = tf.get_variables('lookup_table',[vocab_size,embedding_size],
                                        dtype=dtype,initializer = initializer)
    return tf.nn.embedding_lookup(lookup_table,input_ids)


def embedding_lookup_with_gather(input_ids,vocab_size,embedding_size,initializer,
                                scope = 'embedding',reuse = None,dtype = tf.float32):
    with tf.variable_scope(scope,reuse = reuse):
        lookup_table = tf.get_variable('lookup_table',[vocab_size,embedding_size],
                                        initializer=initializer,reuse = reuse)
        flat_input_ids = tf.reshape(input_ids,[-1])
        output = tf.gather(lookup_table,flat_input_ids)
        output = tf.reshape(output,[input_ids.shape.as_list[0],input_ids.shape.as_list[1],embedding_size])
        return output

def positional_embedding(pos_seq,inv_freq):
    pos_seq  = tf.expand_dims(pos_seq,0)
    inv_freq = tf.expand_dims(inv_freq,1)
    sin_cos_embedding = tf.concat([tf.sin(pos_seq),tf.cos(pos_seq)],-1)

    return sin_cos_embedding


def positional_encoding(max_length,embedding_size):
    freq_seq = tf.range(0,embedding_size,2.0)
    inv_freq = 1 / (10000**(freq_seq/embedding_size))

    pos_seq = tf.range(0,max_length)
    pos_embedding = positional_embedding(pos_seq,inv_freq)


def attention_layer(input_tensor,
                    num_attention_head,
                    attn_head_size):
    """
    input_tensor [B,N,S] 

    num_attention_head : K
    attn_head_size : H
    """

    def transpose_for_score(input_tensor,batch_size,seq_length,num_attn_head,attn_head_size):
        input_tensor = tf.reshape(input_tensor,[batch_size,seq_length,
                                                num_attn_head,attn_head_size])
        output_tensor = tf.transpose(input_tensor,[0,2,1,3])
        return output_tensor


    input_shape = input_tensor.shape.as_list()
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    hidden_size = input_shape[2]
    #[B*N,S]
    input_tensor = tf.reshape(input_tensor,[-1,input_shape[-1]])

    #[B*N,K*H]
    query = tf.layers.dense(
        input_tensor,
        num_attention_head * attn_head_size,
        name = 'query'
        )
    #[B*N,K*H]
    key = tf.layers.dense(
        input_tensor,
        num_attenion_head*attn_head_size,
        name = 'key'
        )
    #[B*N,K*H]
    value =tf.layers.dense(
        input_tensor,
        num_attenion_head*attn_head_size,
        name = 'value'
        ) 
    
    #[B,K,N,H]
    query = transpose_for_score(query,batch_size,seq_length,
                                num_attn_head,attn_head_size)
    key = transpose_for_score(key,batch_size,seq_length,
                                num_attn_head,attn_head_size)
    attention_score = tf.matmul(query,key,transpose_b = True)
    attention_score = tf.multiply(attention_score,1.0/math.sqrt(attn_head_size))
    attention_score = tf.softmax(attention_score,axis = -1)
    