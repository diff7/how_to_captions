# IMPORTS

import os
import json
import tensorflow as tf
import time
import numpy as np
from tokenizers import SentencePieceBPETokenizer

class Looper:
    def __init__(self,
            encoder,
            decoder,
            extractor):


        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.encoder_class = encoder
        self.encoder = None
        self.decoder_class = decoder
        self.decoder = None
        self.extractor = extractor

        self.params = {}
        self.params  = self.set_params()
        self.tokenizer = None


    def set_params(self, params=None, verobese=True):
        if params is None:
            self.params['VOCAB_SIZE'] = 5000
            self.params['BATCH_SIZE'] = 32
            self.params['BUFFER_SIZE'] = 5000       # for shuffle
            self.params['embedding_dim'] = 512
            self.params['embedding_words'] = 300
            self.params['units'] = 512     # gru units
            self.params['embedding_size'] = self.params['VOCAB_SIZE'] + 1
            self.params['MAX_LENGTH'] = 20      # max len of tokens to train
            self.params['TOKENIZER_FOLDER'] = './tokenizer/'
            self.params['TOKENIZER_NAME'] = 'spbe_tokenizer.e'
            self.params['CHECKPOINT_FOLDER'] = 'saved_models'

        else:
            self.params.update(params)

        self.encoder = self.encoder_class(self.params['embedding_dim'])
        self.decoder = self.decoder_class(self.params['embedding_words'],
                                    self.params['units'],
                                    self.params['embedding_size'])

        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                           decoder = self.decoder,
                           optimizer = self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                    self.params['CHECKPOINT_FOLDER'],
                                                    max_to_keep = 5)

        if verobese:
            print('Training with the following params:')
            for key in self.params:
                print(f"{key} : {self.params[key]}")
            print('## Run set_params to change params ##')

        return self.params

    def load_last_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            print(f'load from {self.ckpt_manager.latest_checkpoint}')
            # restoring the latest checkpoint in checkpoint_path
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def load_tokenizer(self):
        path = os.path.join(self.params['TOKENIZER_FOLDER'],self.params['TOKENIZER_NAME'])
        print(f'loading from {path}')
        sbpe_tokenizer = SentencePieceBPETokenizer(path+'-vocab.json',path+'-merges.txt')
        sbpe_tokenizer.enable_padding(max_length=self.params['MAX_LENGTH'])
        sbpe_tokenizer.add_special_tokens(['<start>','<end>'])
        self.tokenizer = sbpe_tokenizer
        return sbpe_tokenizer



    def make_tokenizer(self,
                   text_lines,
                   file_name='spbe_tokenizer.e',
                   text_file_name='titles.txt'):

        with open(text_file_name, 'w') as f:
            for t in text_lines:
                f.write(t+'\n')

        sbpe_tokenizer = SentencePieceBPETokenizer()
        sbpe_tokenizer.train(files=text_file_name, vocab_size=self.params['VOCAB_SIZE']-2)
        sbpe_tokenizer.enable_padding(max_length=self.params['MAX_LENGTH'])
        sbpe_tokenizer.add_special_tokens(['<start>','<end>'])
        sbpe_tokenizer.save(TOKENIZER_FOLDER,file_name)
        return sbpe_tokenizer

    def make_dataset(self,
                    images,
                    captions,
                    loader_type='normal'):

        if self.tokenizer is None:
            print('Load or train tokenizer')
            return None

        out = self.tokenizer.encode_batch(['<start> '+t+' <end>'  for t in captions])
        encoded_list= [t.ids for t in out]

        captions_filtered = []
        images_filtered = []

        for i,t in enumerate(encoded_list):
            if len(t)<=self.params['MAX_LENGTH']:
                captions_filtered.append(t)
                images_filtered.append(images[i])

        images = images_filtered

        print('max title len',max(len(t) for t in captions_filtered))
        print(f'len titles: {len(captions_filtered)},len images: {len(images)}')
        encoded_captios =  tf.convert_to_tensor(captions_filtered)
        print(f'captions vector shape {encoded_captios.shape}')

        if loader_type =='normal':
            loader = self.load_image
            print('NOT using augmentations in loader')
        if loader_type =='aug':
            print('Using augmentations in loader')
            loader = self.load_image_aug

        # Load the numpy files
        def map_func(img_name, cap):
            #img_tensor = np.load(img_name.decode('utf-8')+'.npy')
            img_tensor = loader(img_name)
            return img_tensor, cap

        dataset = tf.data.Dataset.from_tensor_slices((images, encoded_captios))
        # Use map to load the numpy files in parallel
        dataset = dataset.map(map_func)

        # Shuffle and batch
        dataset = dataset.shuffle(self.params['BUFFER_SIZE']).batch(self.params['BATCH_SIZE'])
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


    def loss_function(self, real, pred):  # ZERO PADS <pad> idx - 0
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, image_array, encoded_captions):
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=encoded_captions.shape[0])
        dec_input = tf.expand_dims([self.tokenizer.token_to_id('<start>')] * encoded_captions.shape[0], 1)

        with tf.GradientTape() as tape:
            features_one, features_two = self.get_encoded_features(image_array)
            for i in range(1, encoded_captions.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features_one, features_two, hidden)
                loss += self.loss_function(encoded_captions[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(encoded_captions[:, i], 1)

        total_loss = (loss / int(encoded_captions.shape[1]))
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss


    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.image.random_flip_left_right(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img #,image_path

    def load_image_aug(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 10)
        if bool(np.random.randint(2)):
            img = img +  tf.keras.backend.random_normal((299,299,3),0,np.random.randint(20))
        if bool(np.random.randint(2)):
            img = tf.image.random_contrast(img, 0, 2)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img #, image_path


    def predict_one(self,
                    image_path,
                    t,           # temeperature for randomness of text generation
                    argmax = 2,  # not choose argmax prediction each k times
                    seed=None):  # predict a sentence starting with some word
        result = []
        score = 0
        hidden = self.decoder.reset_state(batch_size=1)
        image_array = tf.expand_dims(self.load_image(image_path), 0)
        features_one, features_two = self.get_encoded_features(image_array)

        if (seed is not None) and (self.tokenizer.token_to_id(seed) is not None):
            print('seed :', seed)
            dec_input = tf.expand_dims([self.tokenizer.token_to_id(seed)], 0)
            result.append(self.tokenizer.token_to_id(seed))
        else:
            #print('seed is None')
            dec_input = tf.expand_dims([self.tokenizer.token_to_id('<start>')], 0)

        for i in range(self.params['MAX_LENGTH']):
            predictions, hidden, _ = self.decoder(dec_input, features_one, features_two, hidden)
            ### Slighlty randomize prediction - can be changed to Beam Search ###
            # SET TO 2 - 4 IF ARGMAX POlICY REQUIRED #
            if i%argmax==0:
                #predictions = tf.nn.softmax(predictions/t)
                predicted_id = tf.random.categorical(predictions/t, 1)[0][0].numpy()
            else:
                predicted_id = int(tf.argmax(predictions[0]).numpy())
            result.append(predicted_id)

            if self.tokenizer.id_to_token(predicted_id) == '<end>':
                return result, score/len(result)

            dec_input = tf.expand_dims([predicted_id], 0)
            score+=predictions[0][predicted_id].numpy()

        return result, score/len(result)

    @tf.function
    def get_encoded_features(self, image_array):
        batch_features_one, batch_features_two  = self.extractor(image_array)
        batch_features_one = tf.reshape(batch_features_one,
                                        (batch_features_one.shape[0], -1, batch_features_one.shape[3]))
        batch_features_two = tf.reshape(batch_features_two,
                                    (batch_features_two.shape[0], -1, batch_features_two.shape[3]))
        features_one, features_two = self.encoder(batch_features_one, batch_features_two)
        return features_one, features_two

    @tf.function
    def predict_step_batch(self, image_array, curr_batch_size, t=0.5, argmax=1, at_k=5):

        # initiate variables
        pred_ids_all = []
        pred_ids = tf.convert_to_tensor([self.tokenizer.token_to_id('<start>')]*curr_batch_size)
        pred_ids = tf.cast(pred_ids, tf.int64)
        pred_ids_all.append(pred_ids)
        dec_input = tf.expand_dims(pred_ids, 1)

        hidden = self.decoder.reset_state(batch_size=curr_batch_size)
        features_one, features_two = self.get_encoded_features(image_array)

        for i in range(1, at_k):
            # passing the features through the decoder
            predictions, hidden, _ = self.decoder(dec_input, features_one, features_two, hidden)
            if i%argmax==0:
                pred_ids = tf.random.categorical(predictions/t, 1)[:,0] #.numpy()
            else:
                # TODO rewrite to TF not np
                #print('arg')
                pred_ids =  tf.math.argmax(predictions, axis=1)

            pred_ids_all.append(pred_ids)

            dec_input = tf.expand_dims(pred_ids, 1)

        return tf.stack(pred_ids_all, axis=1)

    def precicion_atk_batch(self, dataset, at_k,t, argmax=2):
        score = 0
        for i , (image_array, encoded_captions) in enumerate(dataset):
            curr_batch_size = encoded_captions.shape[0]
            pred_words = self.predict_step_batch(image_array, curr_batch_size, t=t, at_k=at_k, argmax=argmax)
            mask = tf.cast(tf.ones(shape=(curr_batch_size, at_k)),tf.int64)

            # zero special token words & some frequen words
            for zero_token in ['<start>','<end>','▁how','▁to']:
                mask *=tf.cast(tf.not_equal(pred_words,self.tokenizer.token_to_id(zero_token)),tf.int64)

            pred_words*=mask
            true = tf.cast(encoded_captions[:,:at_k], tf.int64)
            score += (tf.reduce_sum(tf.cast(pred_words == true, tf.int8)) / curr_batch_size)
        return score.numpy()/(i+1)



    def train(self, train, val, num_epochs, save_n=3):
        for epoch in range(0, num_epochs):
            start = time.time()
            total_loss = 0

            for (batch, (image_array, encoded_captions)) in enumerate(train):
                batch_loss, t_loss = self.train_step(image_array, encoded_captions)
                total_loss += t_loss
                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                      epoch + 1, batch, batch_loss.numpy() / int(encoded_captions.shape[0])))
            # storing the epoch end loss value to plot later

            # TODO add tensorboard

            if epoch % save_n == 0:
                train_score = self.precicion_atk_batch(train, at_k=6,t=0.5, argmax=2)
                val_score = self.precicion_atk_batch(val, at_k=6,t=0.5, argmax=2)
                print(f'precision at | val: {val_score}, train: {train_score}')
                self.ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                                 total_loss/(batch+1)))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
