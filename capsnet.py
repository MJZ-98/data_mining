import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule,channels, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        if(self.channels!=0):
            assert int(self.input_num_capsule/self.channels)/(self.input_num_capsule/self.channels)==1, "error"
            self.W = self.add_weight(shape=[self.num_capsule, self.channels,
                                            self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')

            self.B = self.add_weight(shape=[self.num_capsule,self.dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='B')
        else:
            self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                            self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')
            self.B = self.add_weight(shape=[self.num_capsule,self.dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='B')


        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)

        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        if(self.channels!=0):
            W2 = K.repeat_elements(self.W,int(self.input_num_capsule/self.channels),1)
        else:
            W2 = self.W

        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, W2, [2, 3]) , elems=inputs_tiled)

        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):

            c = tf.nn.softmax(b, dim=1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2])+ self.B)

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

import keras.callbacks as callbacks
from keras.callbacks import Callback
import numpy as np
import os

class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.
    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).
    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.
    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save(filepath, overwrite=True)


class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.
    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr, save_dir):
        """
        Initialize a snapshot callback builder.
        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.save_dir = save_dir

    def get_callbacks(self,log, model_prefix='Model'):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.
        Args:
            model_prefix: prefix for the filename of the weights.
        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists(self.save_dir+'/weights/'):
            os.makedirs(self.save_dir+'/weights/')

        callback_list = [callbacks.ModelCheckpoint(self.save_dir+"/weights/weights_{epoch:002d}.h5", monitor="val_capsnet_acc",
                                                   save_best_only=True, save_weights_only=False),
                         callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix=self.save_dir+'/weights/%s' % model_prefix), log]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

    import keras
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils.vis_utils import plot_model

from utils import combine_images, load_emnist_balanced
from PIL import Image, ImageFilter
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from snapshot import SnapshotCallbackBuilder
import os
import numpy as np
import tensorflow as tf
import os
import argparse

# ----------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.set_image_data_format('channels_last')

"""
Switching the GPU to allow growth
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


def CapsNet(input_shape, n_class, routings):
    """
    Defining the CapsNet
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
    """
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)
    primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,channels=32,name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    """
    Decoder Network
    """
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)
    C=input_shape[2]
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=16*n_class, activation="relu", output_dim=7*7*32*C))
    decoder.add(Reshape((7, 7, 32*C)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(layers.Deconvolution2D(32*C, 3, 3,subsample=(1, 1),border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(16*C, 3, 3,subsample=(2, 2),border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(8*C, 3, 3,subsample=(2, 2),border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(4*C, 3, 3,subsample=(1, 1),border_mode='same', activation="relu"))
    decoder.add(layers.Deconvolution2D(1*C, 3, 3,subsample=(1, 1),border_mode='same', activation="sigmoid"))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    """
    Models for training and evaluation (prediction)
    """
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Marginal loss used for the CapsNet training
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        shuffle = True,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=snapshot.get_callbacks(log,model_prefix=model_prefix))

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    """
    Testing the trained CapsuleNet
    """
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size*8)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/float(y_test.shape[0]))

class dataGeneration():
    def __init__(self, model,data,args,samples_to_generate = 2):
        """
        Generating new images
        :param model: the pre-trained CapsNet model
        :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
        :param args: arguments
        :param samples_to_generate: number of new training samples to generate per class
        """
        self.model = model
        self.data = data
        self.args = args
        self.samples_to_generate = samples_to_generate
        print("-"*100)

        (x_train, y_train), (x_test, y_test), x_recon = self.remove_missclassifications()
        self.data = (x_train, y_train), (x_test, y_test)
        self.reconstructions = x_recon
        self.inst_parameter, self.global_position, self.masked_inst_parameter = self.get_inst_parameters()
        print("Instantiation parameters extracted.")
        print("-"*100)
        self.x_decoder_retrain,self.y_decoder_retrain = self.decoder_retraining_dataset()
        self.retrained_decoder = self.decoder_retraining()
        print("Decoder re-training completed.")
        print("-"*100)
        self.class_variance, self.class_max, self.class_min = self.get_limits()
        self.generated_images,self.generated_labels = self.generate_data()
        print("New images of the shape ",self.generated_images.shape," Generated.")
        print("-"*100)

    def save_output_image(self,samples,image_name):
        """
        Visualizing and saving images in the .png format
        :param samples: images to be visualized
        :param image_name: name of the saved .png file
        """
        if not os.path.exists(args.save_dir+"/images"):
            os.makedirs(args.save_dir+"/images")
        img = combine_images(samples)
        img = img * 255
        Image.fromarray(img.astype(np.uint8)).save(args.save_dir + "/images/"+image_name+".png")
        print(image_name, "Image saved.")

    def remove_missclassifications(self):
        """
        Removing the wrongly classified samples from the training set. We do not alter the testing set.
        :return: dataset with miss classified samples removed and the initial reconstructions.
        """
        model = self.model
        data = self.data
        args = self.args
        (x_train, y_train), (x_test, y_test) = data
        y_pred, x_recon = model.predict(x_train, batch_size=args.batch_size)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_train, 1))/y_train.shape[0]
        cmp = np.argmax(y_pred, 1) == np.argmax(y_train, 1)
        bin_cmp = np.where(cmp == 0)[0]
        x_train = np.delete(x_train,bin_cmp,axis=0)
        y_train = np.delete(y_train,bin_cmp,axis=0)
        x_recon = np.delete(x_recon,bin_cmp,axis=0)
        self.save_output_image(x_train[:100],"original training")
        self.save_output_image(x_recon[:100],"original reconstruction")
        return (x_train, y_train), (x_test, y_test), x_recon


    def get_inst_parameters(self):
        """
        Extracting the instantiation parameters for the existing training set
        :return: instantiation parameters, corresponding labels and the masked instantiation parameters
        """
        model = self.model
        data = self.data
        args = self.args
        (x_train, y_train), (x_test, y_test) = data
        if not os.path.exists(args.save_dir+"/check"):
            os.makedirs(args.save_dir+"/check")

        if not os.path.exists(args.save_dir+"/check/x_inst.npy"):
            get_digitcaps_output = K.function([model.layers[0].input],[model.get_layer("digitcaps").output])

            get_capsnet_output = K.function([model.layers[0].input],[model.get_layer("capsnet").output])

            if (x_train.shape[0]%args.num_cls==0):
                lim = int(x_train.shape[0]/args.num_cls)
            else:
                lim = int(x_train.shape[0]/args.num_cls)+1

            for t in range(0,lim):
                if (t==int(x_train.shape[0]/args.num_cls)):
                    mod = x_train.shape[0]%args.num_cls
                    digitcaps_output = get_digitcaps_output([x_train[t*args.num_cls:t*args.num_cls+mod]])[0]
                    capsnet_output = get_capsnet_output([x_train[t*args.num_cls:t*args.num_cls+mod]])[0]
                else:
                    digitcaps_output = get_digitcaps_output([x_train[t*args.num_cls:(t+1)*args.num_cls]])[0]
                    capsnet_output = get_capsnet_output([x_train[t*args.num_cls:(t+1)*args.num_cls]])[0]
                masked_inst = []
                inst = []
                where = []
                for j in range(0,digitcaps_output.shape[0]):
                    ind = capsnet_output[j].argmax()
                    inst.append(digitcaps_output[j][ind])
                    where.append(ind)
                    for z in range(0,args.num_cls):
                        if (z==ind):
                            continue
                        else:
                            digitcaps_output[j][z] = digitcaps_output[j][z].fill(0.0)
                    masked_inst.append(digitcaps_output[j].flatten())
                masked_inst = np.asarray(masked_inst)
                masked_inst[np.isnan(masked_inst)] = 0
                inst = np.asarray(inst)
                where = np.asarray(where)
                if (t==0):
                    x_inst = np.concatenate([inst])
                    pos = np.concatenate([where])
                    x_masked_inst = np.concatenate([masked_inst])
                else:
                    x_inst = np.concatenate([x_inst,inst])
                    pos = np.concatenate([pos,where])
                    x_masked_inst = np.concatenate([x_masked_inst,masked_inst])
            np.save(args.save_dir+"/check/x_inst",x_inst)
            np.save(args.save_dir+"/check/pos",pos)
            np.save(args.save_dir+"/check/x_masked_inst",x_masked_inst)
        else:
            x_inst = np.load(args.save_dir+"/check/x_inst.npy")
            pos = np.load(args.save_dir+"/check/pos.npy")
            x_masked_inst = np.load(args.save_dir+"/check/x_masked_inst.npy")
        return x_inst,pos,x_masked_inst

    def decoder_retraining_dataset(self):
        """
        Generating the dataset for the decoder retraining technique with unsharp masking
        :return: training samples and labels for decoder retraining
        """
        model = self.model
        data = self.data
        args = self.args
        x_recon = self.reconstructions
        (x_train, y_train), (x_test, y_test) = data
        if not os.path.exists(args.save_dir+"/check"):
            os.makedirs(args.save_dir+"/check")
        if not os.path.exists(args.save_dir+"/check/x_decoder_retrain.npy"):
            for q in range(0,x_recon.shape[0]):
                save_img = Image.fromarray((x_recon[q]*255).reshape(28,28).astype(np.uint8))
                image_more_sharp = save_img.filter(ImageFilter.UnsharpMask(radius=1, percent=1000, threshold=1))
                img_arr = np.asarray(image_more_sharp)
                img_arr = img_arr.reshape(-1,28,28,1).astype('float32') / 255.
                if (q==0):
                    x_recon_sharped = np.concatenate([img_arr])
                else:
                    x_recon_sharped = np.concatenate([x_recon_sharped,img_arr])
            self.save_output_image(x_recon_sharped[:100],"sharpened reconstructions")
            x_decoder_retrain = self.masked_inst_parameter
            y_decoder_retrain = x_recon_sharped
            np.save(args.save_dir+"/check/x_decoder_retrain",x_decoder_retrain)
            np.save(args.save_dir+"/check/y_decoder_retrain",y_decoder_retrain)
        else:
            x_decoder_retrain = np.load(args.save_dir+"/check/x_decoder_retrain.npy")
            y_decoder_retrain = np.load(args.save_dir+"/check/y_decoder_retrain.npy")
        return x_decoder_retrain,y_decoder_retrain

    def decoder_retraining(self):
        """
        The decoder retraining technique to give the sharpening ability to the decoder
        :return: the retrained decoder
        """
        model = self.model
        data = self.data
        args = self.args
        x_decoder_retrain, y_decoder_retrain = self.x_decoder_retrain,self.y_decoder_retrain

        decoder = eval_model.get_layer('decoder')
        decoder_in = layers.Input(shape=(16*47,))
        decoder_out = decoder(decoder_in)
        retrained_decoder = models.Model(decoder_in,decoder_out)
        if (args.verbose):
            retrained_decoder.summary()
        retrained_decoder.compile(optimizer=optimizers.Adam(lr=args.lr),loss='mse',loss_weights=[1.0])
        if not os.path.exists(args.save_dir+"/retrained_decoder.h5"):
            retrained_decoder.fit(x_decoder_retrain, y_decoder_retrain, batch_size=args.batch_size, epochs=20)
            retrained_decoder.save_weights(args.save_dir + '/retrained_decoder.h5')
        else:
            retrained_decoder.load_weights(args.save_dir + '/retrained_decoder.h5')

        retrained_reconstructions = retrained_decoder.predict(x_decoder_retrain, batch_size=args.batch_size)
        self.save_output_image(retrained_reconstructions[:100],"retrained reconstructions")
        return retrained_decoder

    def get_limits(self):
        """
        Calculating the boundaries of the instantiation parameter distributions
        :return: instantiation parameter indices in the descending order of variance, min and max values per class
        """
        args = self.args
        x_inst = self.inst_parameter
        pos = self.global_position
        glob_min = np.amin(x_inst.transpose(),axis=1)
        glob_max = np.amax(x_inst.transpose(),axis=1)

        if not os.path.exists(args.save_dir+"/check"):
            os.makedirs(args.save_dir+"/check")
        if not os.path.exists(args.save_dir+"/check/class_cov.npy"):
            for cl in range(0,self.args.num_cls):
                tmp_glob = []
                for it in range(0,x_inst.shape[0]):
                    if (pos[it]==cl):
                        tmp_glob.append(x_inst[it])
                tmp_glob = np.asarray(tmp_glob)
                tmp_glob = tmp_glob.transpose()
                tmp_cov_max = np.flip(np.argsort(np.around(np.cov(tmp_glob),5).diagonal()),axis=0)
                tmp_min = np.amin(tmp_glob,axis=1)
                tmp_max = np.amax(tmp_glob,axis=1)
                if (cl==0):
                    class_cov = np.vstack([tmp_cov_max])
                    class_min = np.vstack([tmp_min])
                    class_max = np.vstack([tmp_max])
                else:
                    class_cov = np.vstack([class_cov,tmp_cov_max])
                    class_min = np.vstack([class_min,tmp_min])
                    class_max = np.vstack([class_max,tmp_max])
            np.save(args.save_dir+"/check/class_cov",class_cov)
            np.save(args.save_dir+"/check/class_min",class_min)
            np.save(args.save_dir+"/check/class_max",class_max)
        else:
            class_cov = np.load(args.save_dir+"/check/class_cov.npy")
            class_min = np.load(args.save_dir+"/check/class_min.npy")
            class_max = np.load(args.save_dir+"/check/class_max.npy")
        return class_cov,class_max,class_min

    def generate_data(self):
        """
        Generating new images and samples with the data generation technique
        :return: the newly generated images and labels
        """
        data = self.data
        args = self.args
        (x_train, y_train), (x_test, y_test) = data
        x_masked_inst = self.masked_inst_parameter
        pos = self.global_position
        retrained_decoder = self.retrained_decoder
        class_cov = self.class_variance
        class_max = self.class_max
        class_min = self.class_min
        samples_to_generate = self.samples_to_generate
        generated_images = np.empty([0,x_train.shape[1],x_train.shape[2],x_train.shape[3]])
        generated_images_with_ori = np.empty([0,x_train.shape[1],x_train.shape[2],x_train.shape[3]])
        generated_labels = np.empty([0])
        for cl in range(0,args.num_cls):
            count = 0
            for it in range(0,x_masked_inst.shape[0]):
                if (count==samples_to_generate):
                    break
                if (pos[it]==cl):
                    count = count + 1
                    generated_images_with_ori = np.concatenate([generated_images_with_ori,x_train[it].reshape(1,x_train.shape[1],x_train.shape[2],x_train.shape[3])])
                    noise_vec = x_masked_inst[it][x_masked_inst[it].nonzero()]
                    for inst in range(int(class_cov.shape[1]/2)):
                        ind = np.where(class_cov[cl]==inst)[0][0]
                        noise = np.random.uniform(class_min[cl][ind],class_max[cl][ind])
                        noise_vec[ind] = noise
                    x_masked_inst[it][x_masked_inst[it].nonzero()] = noise_vec
                    new_image = retrained_decoder.predict(x_masked_inst[it].reshape(1,args.num_cls*class_cov.shape[1]))
                    generated_images = np.concatenate([generated_images,new_image])
                    generated_labels = np.concatenate([generated_labels,np.asarray([cl])])
                    generated_images_with_ori = np.concatenate([generated_images_with_ori,new_image])
        self.save_output_image(generated_images,"generated_images")
        self.save_output_image(generated_images_with_ori,"generated_images with originals")
        generated_labels = keras.utils.to_categorical(generated_labels, num_classes=args.num_cls)
        if not os.path.exists(args.save_dir+"/generated_data"):
            os.makedirs(args.save_dir+"/generated_data")
        np.save(args.save_dir+"/generated_data/generated_images",generated_images)
        np.save(args.save_dir+"/generated_data/generated_label",generated_labels)
        return generated_images,generated_labels

if __name__ == "__main__":
    """
    Setting the hyper-parameters
    """
    parser = argparse.ArgumentParser(description="TextCaps")
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--cnt', default=200, type=int)
    parser.add_argument('-n','--num_cls', default=47, type=int, help="Iterations")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--samples_to_generate', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--save_dir', default='./emnist_bal_200')
    parser.add_argument('-dg', '--data_generate', action='store_true',
                        help="Generate new data with pre-trained model")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_emnist_balanced(args.cnt)
    #(x_train, y_train), (x_test, y_test) = load_my_data(args.cnt)
    # (x_train, y_train), (x_test, y_test) = load_li_data()

    print('-------------/n-------------/n-----',(x_train),len(x_test))

    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                routings=args.routings)
    if (args.verbose):
        model.summary()

    """
    Snap shot training
    :param M: number of snapshots
    :param nb_epoch: number of epochs
    :param alpha_zero: initial learning rate
    """
    M = 3
    nb_epoch = T = args.epochs
    alpha_zero = 0.01
    model_prefix = 'Model_'
    snapshot = SnapshotCallbackBuilder(T, M, alpha_zero,args.save_dir)

    if args.weights is not None:
        model.load_weights(args.weights)
    if not args.data_generate:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        test(model=eval_model, data=(x_test, y_test), args=args)

    else:
        if args.weights is None:
            print('No weights are provided. You need to train a model first.')
        else:
            data_generator = dataGeneration(model=eval_model, data=((x_train, y_train), (x_test, y_test)), args=args, samples_to_generate = args.samples_to_generate)


import numpy as np
import math

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def load_emnist_balanced(cnt):
    from scipy import io as spio
    from keras.utils import to_categorical
    import numpy as np
    emnist = spio.loadmat("data/matlab/emnist-balanced.mat")

    print(emnist['dataset'])

    classes = 47
    cnt = cnt
    lim_train = cnt*classes

    x_train = emnist["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)
    y_train = emnist["dataset"][0][0][0][0][0][1]
    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    y_test = emnist["dataset"][0][0][1][0][0][1]

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order="A").astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="A").astype('float32') / 255.

    y_train = (y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    #Append equal number of training samples from each class to x_train and y_train
    x_tr = []
    y_tr = []
    count = [0] * classes
    for i in range(0,x_train.shape[0]):
        if (sum(count)==classes*cnt):
            break
        name = (y_train[i])
        if (count[int(name)]>=cnt):
            continue
        count[int(name)] = count[int(name)]+1
        x_tr.append(x_train[i])
        y_tr.append(name)
    x_tr = np.asarray(x_tr)
    y_tr = np.asarray(y_tr)
    y_tr = to_categorical(y_tr.astype('float32'))

    print(x_tr.shape,y_tr.shape,x_test.shape,y_test.shape)
    return (x_tr, y_tr), (x_test, y_test)