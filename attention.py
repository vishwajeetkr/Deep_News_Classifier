import tensorflow as tf


def attention(hidden_vectors, context_vectors):
    
    # shapes : hidden_vectors = (None, 30, 100); context_vectors = (None, 100, 100)
    # 
    dimensions = []
    #print("a")
    
    #print(hidden_vectors.get_shape()[0].value)
    #print(hidden_vectors.get_shape()[2].value)
    #tf.shape(x)[0]
    dimensions.append(tf.shape(hidden_vectors)[0])
    
    dimensions.append(hidden_vectors.get_shape()[1].value)
    dimensions.append(hidden_vectors.get_shape()[2].value)
    
    embedding_length = hidden_vectors.get_shape()[2].value
    reshape_hidden_vectors = tf.reshape(hidden_vectors, [tf.shape(hidden_vectors)[0]*tf.shape(hidden_vectors)[1], tf.shape(hidden_vectors)[2]])
    # current shape: hidden_vectors = (None*30, 100)
        
    W_a = tf.Variable(tf.random_normal([embedding_length, embedding_length], stddev=0.1))
        
    temp = tf.matmul(reshape_hidden_vectors, W_a)
    # current shape: temp = (None*30, 100)
    #print("passed stage 1")
    temp = tf.reshape(temp, [dimensions[0], dimensions[1], dimensions[2]])
    # current shape: temp = (None, 30, 100)
    
    context_vectors_transpose = tf.transpose(context_vectors, perm=[0, 2, 1])
    # current shape: temp = (None, 100, 100)
    #print("passed stage 2")
    
    attention_weigths = tf.matmul(temp, context_vectors_transpose)
    # shape attention_weigths: (None, 30, 100)
    
    attention_probability = tf.nn.softmax(attention_weigths)
    # shape attention_probability: (None, 30, 100)
    
    c_attention_vectors = tf.matmul(attention_probability, context_vectors)
    #print(c_attention_vectors.get_shape()[0], c_attention_vectors.get_shape()[1], c_attention_vectors.get_shape()[2])
    # shape c_attention_vectors: (None, 30, 100)
    #print(tf.shape(c_attention_vectors), tf.shape(hidden_vectors))
    
    concatnated_vectors = tf.concat((c_attention_vectors, hidden_vectors), 2)
    #print(concatnated_vectors.get_shape()[0], concatnated_vectors.get_shape()[1], concatnated_vectors.get_shape()[2])
    W_c = tf.Variable(tf.random_normal([2*embedding_length, embedding_length], stddev=0.1))
    
    reshape_concatnated_vectors = tf.reshape(concatnated_vectors, [tf.shape(concatnated_vectors)[0]*concatnated_vectors.get_shape()[1], concatnated_vectors.get_shape()[2]])
    #print(reshape_concatnated_vectors.get_shape()[0], reshape_concatnated_vectors.get_shape()[1])
    
    final_vectors = tf.nn.tanh(tf.matmul(reshape_concatnated_vectors, W_c))
    #print(final_vectors.get_shape()[0], final_vectors.get_shape()[1])
    reshaped_final_vectors = tf.reshape(final_vectors, [tf.shape(c_attention_vectors)[0], c_attention_vectors.get_shape()[1], c_attention_vectors.get_shape()[2]])
    #print(tf.shape(reshaped_final_vectors)[0], tf.shape(reshaped_final_vectors)[1], tf.shape(reshaped_final_vectors)[2])
    #print("passed stage 3")
    #print(reshaped_final_vectors.get_shape()[0], reshaped_final_vectors.get_shape()[1], reshaped_final_vectors.get_shape()[2])
    #print("something")
    return reshaped_final_vectors



