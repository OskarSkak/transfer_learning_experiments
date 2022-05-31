from db.crud import save_diagnostics
from db.models import Diagnostic, FitHistory, GPUUsageTest, GPUUsageTrain
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.nasnet import NASNetLarge
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import optimizers
# from keras.models import Model
# from keras.layers import Dense
# from keras.layers import Flatten
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
# from matplotlib import pyplot
import sys
import time
import GPUtil
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

import tensorflow as tf
from data_utils import DataUtils

import subprocess as sp
import os
from threading import Thread , Timer
import sched, time
# from tf_explain.core import GradCAM, GradientsInputs, VanillaGradients, ExtractActivations, IntegratedGradients, OcclusionSensitivity, SmoothGrad
# from tf_explain_local.tf_explain.core.grad_cam import GradCAM
# from tf_explain_local.tf_explain.core.vanilla_gradients import VanillaGradients
# from tf_explain_local.tf_explain.core.gradients_inputs import GradientsInputs
import numpy as np
import tensorflow.keras.backend as K



# from vis_utils import summarize_diagnostics
# import keras
# import tensorflow as tf


class TransferLearningModel():
    def load_data(self, width=32, height=32):
        self.trainX, self.trainY, self.testX, self.testY = DataUtils().api_load_medical_data(width, height)

    def load_pre_trained_model(self, name):
        print(f'attempting to load pretrained model: {name}')
        if name == 'VGG16':
            print('loading vgg16')
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.trainX[0].shape)
            self.base_model.trainable = False
        elif name == 'Xception':
            print('loading xception')
            self.base_model = Xception(weights='imagenet', include_top=False, input_shape=self.trainX[0].shape)
            self.base_model.trainable = False
        elif name == 'ResNet152V2':
            print('loading resnet')
            self.base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=self.trainX[0].shape)
            self.base_model.trainable = False
        elif name == 'NASNetLarge':
            print('loading nasnet')
            self.base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=self.trainX[0].shape)
            self.base_model.trainable = False
        elif name == 'MobileNetV2':
            print('loading mobilenet')
            self.base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.trainX[0].shape)
            self.base_model.trainable = False
        elif name == 'InceptionResNetV2':
            print('loading InceptionResNetV2')
            self.base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.trainX[0].shape)
            self.base_model.trainable = False
        else:
            print("incorrect model specified...")
    
    def add_fully_connected_layers(self, dense_activation='relu', pred_activation='softmax', dropout_rate = 0):
        flatten_layer = layers.Flatten()
        dense_layer1 = layers.Dense(50, activation=dense_activation)
        dense_layer2 = layers.Dense(20, activation=dense_activation)
        prediction_layer = layers.Dense(3, activation=pred_activation)

        dropout_1 = layers.Dropout(dropout_rate)
        dropout_2 = layers.Dropout(dropout_rate*0.5)

        # self.model = models.Sequential([
        #     self.base_model
        # ])
        self.model = models.Sequential()
        for layer in self.base_model.layers:
            self.model.add(layer)

        self.model.add(flatten_layer)
        self.model.add(dense_layer1)
        if dropout_rate != 0:
            self.model.add(dropout_1)
        self.model.add(dense_layer2)
        if dropout_rate != 0:
            self.model.add(dropout_2)
        self.model.add(prediction_layer)

        # self.model = models.Sequential([
        #     self.base_model,
        #     flatten_layer,
        #     dense_layer1,
        #     dropout_1,
        #     dense_layer2,
        #     dropout_2,
        #     prediction_layer
        # ])

    def print_model(self):
        self.model.summary()
        print('layer names')
        for l in self.model.layers:
            print(l.name)

    def compile(self, epochs=100, batch_size=1, learning_rate=0.0001, momentum=0.0, optimizer='adam'):
        opt = None
        if optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'adagrad':
            opt = optimizers.Adagrad(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt, 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # self.model.compile(
        #     optimizer=optimizer, 
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy'],
        #     learning_rate=learning_rate,
        #     momentum=momentum
        # )

        # We only try one here, as 'restore_best_weights' should 
        #es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10,  restore_best_weights=True)

        # history = self.model.fit(self.trainX, self.trainY, epochs=30, validation_split=0.2, batch_size=32,  verbose=1)
        history = self.model.fit(self.trainX, self.trainY, epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=1)
        
        return history
    
    def save_model(self, name):
        self.model.save('./models/' + name)

    def save_summed_results(self, details_about_model, acc):
        f = open("./CNNs/results/summed_res.txt", "a")
        msg = f"{details_about_model} ,ACCURACY: {acc}"
        f.write(msg)
        f.close()
    
    def create_custom_model(self):
        print("creating custom model...")
    

    def grad_cam(self):
        layer_name = self.model.layers[len(self.model.layers) -1].name
        print(f'visualizing on layer {layer_name}')

        print('\n\n')
        print(f'x: {self.trainX[0].shape}')
        print(f'output layer: {self.model.output_shape}')
        print(f'input layer: {self.model.input_shape}')
        
        
        img = tf.keras.preprocessing.image.load_img('../data/COVID-19_Radiography_Dataset/archive/COVID_all/COVID-1.png', target_size=(100, 100))
        img = tf.keras.preprocessing.image.img_to_array(img)
        print(f'ex: {img.shape}')
        print('\n\n')
        # data = ([self.trainX[0]], None)
        data = ([img], None)
        #img2 = img[..., np.newaxis]
        covid_img_index = 1
        # img2 = K.constant(img, shape=(None, 100, 100, 3))
        explainer = GradCAM()

        # layer = self.model.layers[0].get_layer('block5_conv3')

        grid = explainer.explain(data, self.model, class_index=covid_img_index, layer_name='block5_conv3')
        explainer.save(grid, ".", "test_grad_cam.png")
    
    def custom_grad_cam(self):
        img = tf.keras.preprocessing.image.load_img('../data/COVID-19_Radiography_Dataset/archive/COVID_all/COVID-1.png', target_size=(100, 100))
        img = tf.keras.preprocessing.image.img_to_array(img)

        grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer('block5_conv3').output, self.model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 1]

        output = conv_outputs[0]

        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.zeros(output.shape[0: 2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # cam = cv2.resize(cam.numpy(), (224, 224))
        # cam = np.maximum(cam, 0)
        # heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    

    def get_model(self):
        return self.model
    

    


    def gradients(self):
        layer_name = self.model.layers[len(self.model.layers) -1].name
        print(f'visualizing on layer {layer_name}')

        print('\n\n')
        print(f'x: {self.trainX[0].shape}')
        print(f'output layer: {self.model.output_shape}')
        print(f'input layer: {self.model.input_shape}')
        
        
        img = tf.keras.preprocessing.image.load_img('../data/COVID-19_Radiography_Dataset/archive/COVID_all/COVID-1.png', target_size=(100, 100))
        img = tf.keras.preprocessing.image.img_to_array(img)
        print(f'ex: {img.shape}')
        print('\n\n')
        # data = ([self.trainX[0]], None)
        data = ([img], None)
        #img2 = img[..., np.newaxis]
        covid_img_index = 1
        # img2 = K.constant(img, shape=(None, 100, 100, 3))
        explainer = GradientsInputs()

        # layer = self.model.layers[0].get_layer('block5_conv3')

        grid = explainer.explain(data, self.model, class_index=covid_img_index)
        explainer.save(grid, ".", "test_grad_cam.png")
    
    def vanilla(self):
        layer_name = self.model.layers[len(self.model.layers) -1].name
        print(f'visualizing on layer {layer_name}')

        print('\n\n')
        print(f'x: {self.trainX[0].shape}')
        print(f'output layer: {self.model.output_shape}')
        print(f'input layer: {self.model.input_shape}')
        
        
        img = tf.keras.preprocessing.image.load_img('../data/COVID-19_Radiography_Dataset/archive/COVID_all/COVID-1.png', target_size=(100, 100))
        img = tf.keras.preprocessing.image.img_to_array(img)
        print(f'ex: {img.shape}')
        print('\n\n')
        # data = ([self.trainX[0]], None)
        data = (np.array([img]), None)
        #img2 = img[..., np.newaxis]
        covid_img_index = 1
        # img2 = K.constant(img, shape=(None, 100, 100, 3))
        explainer = VanillaGradients()

        # layer = self.model.layers[0].get_layer('block5_conv3')

        grid = explainer.explain(data, self.model, class_index=covid_img_index)
        explainer.save(grid, ".", "test_grad_van.png")


def get_img_array(img_path, size):
        # `img` is a PIL image of size 299x299
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = tf.keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array
    
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def main():
    model_name = str(sys.argv[1]) # Model                     ***
    batch_size = int(sys.argv[2]) # batch size           ***
    optimizer = str(sys.argv[3]) # optimizer             
    learning_rate = float(sys.argv[4]) # learning rate     ***
    epochs = int(sys.argv[5]) # epochs                   ***
    momentum = float(sys.argv[6]) # momentum               ***
    activation_func = str(sys.argv[7]) #                 ***
    dropout_rate = float(sys.argv[8]) #                    ***

    print('________________________Hyperparameter summary________________________')
    print(f'pret model \t{model_name}')
    print(f'batch size \t{batch_size}')
    print(f'optimizer \t{optimizer}')
    print(f'learning r \t{learning_rate}')
    print(f't epochs \t{epochs}')
    print(f'momentum \t{momentum}')
    print(f'acti func \t{activation_func}')
    print(f'dropout r \t{dropout_rate}')
    print('______________________________________________________________________')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])

    print("loading and preprocessing data...")
    model = TransferLearningModel()
    model.load_data(100, 100)

    print("loading pre-trained model and adding fully connected layers...")
    model.load_pre_trained_model(name=model_name)
    model.add_fully_connected_layers(dense_activation=activation_func, dropout_rate=dropout_rate)
    # model.print_model()

    history = model.compile(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
    

    # print("evaluating model...")
    # _, acc = model.model.evaluate(model.testX, model.testY, verbose=0)

    # train_loss = history.history['loss']
    # val_loss   = history.history['val_loss']
    # train_acc  = history.history['accuracy']
    # val_acc    = history.history['val_accuracy'] 
    # model.vanilla()
    # model.gradients()
    # model.grad_cam()
    img_array = preprocess_input(get_img_array('../data/COVID-19_Radiography_Dataset/archive/COVID_all/COVID-1.png', size=(100, 100)))
    
    local_model = model.get_model()
    local_model.layers[-1].activation = None
    preds = local_model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv3')

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    

if __name__ == '__main__':
    main()


# epoch [10, 30, 60, 100]
# 