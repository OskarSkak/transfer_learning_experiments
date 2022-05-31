from db.crud import save_diagnostics
from db.models import Diagnostic, FitHistory, GPUUsageTest, GPUUsageTrain
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
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

import tensorflow as tf

from data_utils_derm_perc import DataUtils

import subprocess as sp
import os
from threading import Thread , Timer
import sched, time

# from vis_utils import summarize_diagnostics
# import keras
# import tensorflow as tf


class TransferLearningModel():
    def load_data(self, width=32, height=32, percentage=0.9):
        self.trainX, self.trainY, self.testX, self.testY = DataUtils().api_load_medical_data(width, height, percentage)
    
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
        dense_layer1 = layers.Dense(4000, activation=dense_activation)
        dense_layer2 = layers.Dense(2000, activation=dense_activation)
        prediction_layer = layers.Dense(9, activation=pred_activation)

        dropout_1 = layers.Dropout(dropout_rate)
        dropout_2 = layers.Dropout(dropout_rate*0.5)

        self.model = models.Sequential([
            self.base_model
        ])
        self.model.add(flatten_layer)
        self.model.add(dense_layer1)
        if dropout_rate != 0:
            self.model.add(dropout_1)
        self.model.add(dense_layer2)
        if dropout_rate != 0:
            self.model.add(dropout_2)
        self.model.add(prediction_layer)

    def print_model(self):
        self.load_pre_trained_model()
        self.add_fully_connected_layers()
        self.model.summary()

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

        # history = self.model.fit(self.trainX, self.trainY, epochs=30, validation_split=0.2, batch_size=32, callbacks=[es], verbose=1)
        history = self.model.fit(self.trainX, self.trainY, epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=1)
        
        return history
    
    def save_model(self, name):
        self.model.save('./models/' + name)

    def save_summed_results(self, details_about_model, acc):
        f = open("./CNNs/results/summed_res.txt", "a")
        msg = f"{details_about_model} ,ACCURACY: {acc}"
        f.write(msg)
        f.close()

 #nvidia-smi --query-gpu=utilization.gpu --format=csv   
    
def get_gpu_util():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv  "
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

gpu_util_core = GPUtil.getGPUs()[0] 

def get_gpu_load():
    load = gpu_util_core.load
    return load

collecting_train_gpu = True
train_gpu_usage = []
train_gpu_load = []

collecting_test_gpu = True
test_gpu_usage = []
test_gpu_load = []

def record_train_gpu_load():
    if not collecting_train_gpu:
        return
    Timer(5.0, record_train_gpu_load).start()
    train_gpu_load.extend(get_gpu_util())

def record_test_gpu_load():
    if not collecting_test_gpu:
        return
    Timer(5.0, record_test_gpu_load).start()
    test_gpu_load.extend(get_gpu_util())

def record_train_gpu():
    if not collecting_train_gpu:
        return
    Timer(5.0, record_train_gpu).start()
    train_gpu_usage.extend(get_gpu_memory())

def record_test_gpu():
    if not collecting_test_gpu:
        return
    Timer(5.0, record_test_gpu).start()
    test_gpu_usage.extend(get_gpu_memory())

def main():
    # ps_pid = subprocess.Popen(["python", "test.py" ,
    #   model, str(batch_size), optimizer, str(learning_rate),
    #   str(epoch), str(early_stopping_patience,
    #   str(momentum), activation_function, str(dropout_rate))])
    model_name = str(sys.argv[1]) # Model                     ***
    batch_size = int(sys.argv[2]) # batch size           ***
    optimizer = str(sys.argv[3]) # optimizer             
    learning_rate = float(sys.argv[4]) # learning rate     ***
    epochs = int(sys.argv[5]) # epochs                   ***
    momentum = float(sys.argv[6]) # momentum               ***
    activation_func = str(sys.argv[7]) #                 ***
    dropout_rate = float(sys.argv[8]) #                    ***
    percentage_split = float(sys.argv[9])

    print('________________________Hyperparameter summary________________________')
    print(f'pret model \t{model_name}')
    print(f'batch size \t{batch_size}')
    print(f'optimizer \t{optimizer}')
    print(f'learning r \t{learning_rate}')
    print(f't epochs \t{epochs}')
    print(f'momentum \t{momentum}')
    print(f'acti func \t{activation_func}')
    print(f'dropout r \t{dropout_rate}')
    print(f'percentage \t{percentage_split}')
    print('______________________________________________________________________')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])

    print("loading and preprocessing data...")
    model = TransferLearningModel()
    model.load_data(100, 100, percentage_split)

    print("loading pre-trained model and adding fully connected layers...")
    model.load_pre_trained_model(name=model_name)
    model.add_fully_connected_layers(dense_activation=activation_func, dropout_rate=dropout_rate)
    
    print("compiling model...")
    record_train_gpu()
    record_train_gpu_load()
    start_train = time.time()
    history = model.compile(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)
    global collecting_train_gpu
    collecting_train_gpu = False
    train_time = time.time() - start_train

    print("evaluating model...")
    record_test_gpu()
    record_test_gpu_load()
    start_test = time.time()
    _, acc = model.model.evaluate(model.testX, model.testY, verbose=0)
    global collecting_test_gpu
    collecting_test_gpu = False
    test_time = time.time() - start_test

    print("ACCURACY: ", acc)

    print("\n\n########## history ################")
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy'] 
    print("train loss = ", train_loss)
    print("val loss = ", val_loss)
    print("train acc = ", train_acc)
    print("val acc = ", val_acc)
    print('gpu load', test_gpu_load)
    print("\n\n")
    # summarize_diagnostics([1, 2, 3], train_loss, val_loss, train_acc, val_acc, "TEST")
    
    # model.save_model("test")
    # model.save_summed_results(network_name, acc)
    diagnostic = Diagnostic()

    diagnostic.model = model_name
    diagnostic.activation_func = activation_func
    diagnostic.batch_size = batch_size
    diagnostic.dropout_rate = dropout_rate
    diagnostic.epochs = epochs
    diagnostic.learning_rate = learning_rate
    diagnostic.momentum = momentum
    diagnostic.optimizer = optimizer
    diagnostic.train_time = train_time
    diagnostic.test_time = test_time
    diagnostic.test_acc = acc
    diagnostic.train_samples = len(model.trainX)
    diagnostic.test_samples = len(model.testX)
    diagnostic.percentage = percentage_split
    diagnostic.dataset = 'derm'

    gpu_train_diagnostics = []
    gpu_test_diagnostics = []
    fit_histories = []

    for count, (gpu, load) in enumerate(zip(train_gpu_usage, train_gpu_load)):
        gpu_train = GPUUsageTrain()
        gpu_train.time = count*5 # (polled every five seconds)
        gpu_train.usage = gpu
        gpu_train.load = load
        gpu_train_diagnostics.append(gpu_train)
    
    for count, (gpu, load) in enumerate(zip(test_gpu_usage, test_gpu_load)):
        gpu_test = GPUUsageTest()
        gpu_test.time = count*5 # (polled every five seconds)
        gpu_test.usage = gpu
        gpu_test.load = load
        gpu_test_diagnostics.append(gpu_test)
    
    for count, (tl, vl, ta, va) in enumerate(zip(train_loss, val_loss, train_acc, val_acc)):
        hist = FitHistory()
        hist.epoch = count + 1
        hist.train_loss = tl
        hist.val_loss = vl
        hist.train_acc = ta
        hist.val_acc = va
        fit_histories.append(hist)
    
    diagnostic.history = fit_histories
    diagnostic.gpu_test = gpu_test_diagnostics
    diagnostic.gpu_train = gpu_train_diagnostics

    # save_diagnostics(diagnostic)

if __name__ == '__main__':
    main()


# epoch [10, 30, 60, 100]
# 