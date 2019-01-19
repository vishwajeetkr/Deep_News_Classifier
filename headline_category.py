import tensorflow as tf
import numpy as np
import json
import attention

max_headline = 30
max_description = 100
embedding_dim = 100


def get_mini_batch(embedding_dict, input_file, block_number1, block_number2, block_size):
    print("block_number-",block_number1,"--",block_number2)
    global max_headline, max_description, embedding_dim
    num_of_blocks = block_number2 - block_number1 + 1
    heading_mini_batch = np.zeros((block_size*num_of_blocks, max_headline, embedding_dim))
    description_mini_batch = np.zeros((block_size*num_of_blocks, max_description, embedding_dim))
    y_mini_batch = np.zeros((block_size*num_of_blocks, 1))
    #count = 0
    with open(input_file, "r") as fp:
        for k, line in enumerate(fp):
            if k >= block_size*(block_number1-1) and k < block_number2*block_size:
                i = k - block_size*(block_number1-1)
                parts = line.strip().split("\t")
                words = parts[0].strip().split(" ")
                #embedding_list = []
                #for word in words:
                #    embedding_list.append(embedding_dict[word])
                #current_len = len(embedding_list)
                #if len(embedding_list) < 35:
                #    embedding_list.append(embedding_dict["unk"] * (35 - current_len))
                for j in range(max_headline):
                    if j < len(words):
                        heading_mini_batch[i][j] = embedding_dict[words[j]]
                    else:
                        heading_mini_batch[i][j] = embedding_dict["unk"]
                        
                words = parts[1].strip().split(" ")
                        
                for j in range(max_description):
                    if j < len(words):
                        description_mini_batch[i][j] = embedding_dict[words[j]]
                    else:
                        description_mini_batch[i][j] = embedding_dict["unk"]
                
                try:
                    y_mini_batch[i] = int(parts[2].strip())
                except (ValueError):
                    #print("error")
                    y_mini_batch[i] = 3
    return (heading_mini_batch, description_mini_batch, y_mini_batch)

def get_embedding_dict(dict_file):
    with open(dict_file) as f:
        dict_ = json.load(f)
    return dict_

def bidirectional_layer(X, n_neurons):
    with tf.variable_scope("1", default_name="1", reuse=tf.AUTO_REUSE):
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X, dtype=tf.float32)
        #outputs, output_states = tf.nn.dynamic_rnn(cell_fw, X, dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        #output_states = tf.concat(output_states, 2)
    return (outputs, output_states)

def bidirectional_layer_2(X, n_neurons):
    with tf.variable_scope("2", default_name="2", reuse=tf.AUTO_REUSE):
        cell_fw_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
        cell_bw_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw_2, cell_bw_2, X, dtype=tf.float32)
        #outputs, output_states = tf.nn.dynamic_rnn(cell_fw, X, dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        #output_states = tf.concat(output_states, 2)
    return (outputs, output_states)

def run():
    global max_headline, max_description, embedding_dim
    n_neurons_bl_1 = 50
    n_neurons_bl_2 = 25
    n_outputs = 41
    learning_rate = 0.01
    headline = tf.placeholder(tf.float32, [None, max_headline, embedding_dim]) # n_steps = max_len, n_inputs=embedding_dim
    description = tf.placeholder(tf.float32, [None, max_description, embedding_dim])
    y = tf.placeholder(tf.int32, [None, 1])
    
    outputs_headline, states_headline = bidirectional_layer(headline, n_neurons_bl_1)
    outputs_description, states_description = bidirectional_layer(description, n_neurons_bl_1)
    
    attended_headline_vectors = attention.attention(outputs_headline, outputs_description)
    
    final_bd_vector, final_bd_state = bidirectional_layer_2(attended_headline_vectors, n_neurons_bl_2)
    
    dimensions = []
    dimensions.append(tf.shape(final_bd_vector)[0])
    dimensions.append(final_bd_vector.get_shape()[1].value)
    dimensions.append(final_bd_vector.get_shape()[2].value)
    
    final_bd_vector = tf.reshape(final_bd_vector, [dimensions[0], dimensions[1]*dimensions[2]])
    logits = tf.contrib.layers.fully_connected(final_bd_vector, n_outputs, activation_fn=None)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y,[tf.shape(y)[0]]), logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, tf.reshape(y,[tf.shape(y)[0]]), 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    
    embedding_dict = get_embedding_dict("json_embedding_news.json")
    input_file = "dataset_final.txt"
    n_epochs = 10
    batch_size = 1000
    
    init = tf.global_variables_initializer()
    file_output = open("output-15-epoch-lstm.txt","w")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        num_examples = 180950
        num_mini_batches = int(num_examples / batch_size)
        print("num_mini_batch",num_mini_batches)
        for epoch in range(n_epochs):
            length = int(num_mini_batches * 9 / 10)
            for iteration in range(length):
                heading_mini_batch, description_mini_batch, y_mini_batch = get_mini_batch(embedding_dict, input_file, iteration+1, iteration+1, batch_size)
                #outputs_headline_val, states_headline_val, outputs_description_val, states_description_val = sess.run([outputs_headline, states_headline, outputs_description, states_description],feed_dict={headline: heading_mini_batch, description: description_mini_batch})
                sess.run(training_op, feed_dict={headline: heading_mini_batch, description: description_mini_batch, y:y_mini_batch})
                #print("---done---")
                #print(outputs_headline_val.shape)
                #print(states_headline_val[0].shape, states_headline_val[1].shape)
            acc_train = accuracy.eval(feed_dict={headline: heading_mini_batch, description: description_mini_batch, y:y_mini_batch})
            heading_test_batch, description_test_batch, y_test_batch = get_mini_batch(embedding_dict, input_file, length, num_mini_batches, batch_size)
            acc_test = accuracy.eval(feed_dict={headline: heading_test_batch, description: description_test_batch, y:y_test_batch})
            print(epoch, "Train accuracy:", acc_train,", Test accuracy:", acc_test)
            file_output.write(str(epoch)+"Train accuracy:"+str(acc_train)+", Test accuracy:"+str(acc_test)+"\n")
            if epoch % 5 == 4 or epoch == 0:
                save_path = saver.save(sess, "my_model_lstm_"+str(epoch)+".ckpt")
    file_output.close()



if __name__ == "__main__":
    run()

