import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # embedding_size , 512 units
        self.W2 = tf.keras.layers.Dense(units)  # target_len (max_len), 512 units
        self.V = tf.keras.layers.Dense(1)
        self.score = None
        self.attention_weights = None


    def make_features_matrix(self, features):
        self.W_features =  self.W1(features)


    def call(self, features, hidden):
        # features = batch x 64 x 512
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape = batch_size x 16
        # hidden shape == (batch_size, hidden_size)

        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        #self.f1 = self.W1(features)
        #self.f2 = self.W2(hidden_with_time_axis)
        self.score = tf.nn.tanh(self.W_features + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(self.score), axis=1)


        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


## feature extractor

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input

"""
VGG layers - concatenated blocks
40 mixed0, 63 mixed1, 86 mixed2, 100 mixed3, 132 mixed4,
164 mixed5 196 mixed6 ,228 mixed7, 248 mixed8, 276 mixed9_0,
279 mixed9, 307 mixed9_1, 310 mixed10 """

# old value is 164

hidden_layer_mid = image_model.layers[279].output
hidden_layer_last = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, [hidden_layer_mid, hidden_layer_last])
image_features_extract_model.trainable = False


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc_one = tf.keras.layers.Dense(embedding_dim)
        self.fc_two = tf.keras.layers.Dense(embedding_dim)

    def call(self, x_one, x_two):
        x_one = self.fc_one(x_one)
        x_one = tf.nn.relu(x_one)

        x_two = self.fc_two(x_two)
        x_two = tf.nn.relu(x_two)
        return x_one, x_two


class RNN_Decoder_base(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, num_attention=2):
        super(RNN_Decoder_base, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru_one = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.gru_two = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        #self.fc1 = tf.keras.layers.Dense(self.units, )
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attentions = []
        for i in range(num_attention):
            self.attentions.append(BahdanauAttention(self.units))

    def pre_attention(self, features):
        for feature, att in zip(features, self.attentions):
            att.make_features_matrix(feature)


    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class RNN_Decoder_stack_gru(RNN_Decoder_base):
    def call(self, x, features_one, features_two, hidden):
        # features = batch x 64 x 512
        # hidden shape = target.shape[0] == units
        # defining attention as a separate model
        context_vector_one, _ = self.attention_one(features_one, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        emb = self.embedding(x)
        #x = tf.keras.layers.Dropout(0.1)(emb)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector_one, 1), emb], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru_one(x)
        context_vector_two, attention_weights = self.attention_two(features_two, state)
        # TODO #### CHANGE X  TO OUTPUT ERR !!!!
        x = tf.concat([tf.expand_dims(context_vector_two, 1), emb], axis=-1)

        # shape == (batch_size, max_length, hidden_size)
        output, state = self.gru_two(x)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights


class RNN_Decoder_concat(RNN_Decoder_base):
    def call(self, x, features, hidden):
        # features = batch x 64 x 512
        # hidden shape = target.shape[0] == units
        # defining attention as a separate model

        context_vectors = []
        for att, feat in zip(self.attentions, features):
            context_vector_two, attention_weights = att(feat, hidden)
            context_vectors.append(context_vector_two)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)

        emb = self.embedding(x)

        output, state = self.gru_one(emb)
        context_vectors = [tf.cast(context, tf.float32)  for context in context_vectors]
        context_vectors.append(tf.cast(state, tf.float32))
        x = tf.concat(context_vectors, axis=-1)
        output, _ = self.gru_two(tf.expand_dims(x,1))

        #x = tf.concat([output_one, output_one], axis=-1)
        #x = tf.concat([tf.expand_dims(context_vector_two, 1), x], axis=-1)
        # shape == (batch_size, max_length, hidden_size)
        #output, state = self.gru_two(x)

        #x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[-1]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc(output)

        return x, state, attention_weights
