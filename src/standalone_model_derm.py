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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD

# from vis_utils import summarize_diagnostics
# import keras
# import tensorflow as tf        
    
def define_model_one_VGG_block(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 250, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_model_two_VGG_blocks(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 250, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_model_three_VGG_blocks(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 250, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    if dropout_rate:
        model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# define cnn model
def define_model_with_dropout_reg_three_VGG_blocks(drop_out_rate_first_block, drop_out_rate_second_block, drop_out_rate_third_block, learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 250, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(drop_out_rate_first_block))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(drop_out_rate_second_block))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(drop_out_rate_third_block))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def m2tex(model):
    stringlist = []
    model.summary(line_length=70, print_fn=lambda x: stringlist.append(x))
    del stringlist[1:-4:2]
    del stringlist[-1]
    for ix in range(1,len(stringlist)-3):
        tmp = stringlist[ix]
        stringlist[ix] = tmp[0:31]+"& "+tmp[31:59]+"& "+tmp[59:]+"\\\\ \hline"
    stringlist[0] = "Model: test \\\\ \hline"
    stringlist[1] = stringlist[1]+" \hline"
    stringlist[-4] = stringlist[-4]+" \hline"
    stringlist[-3] = stringlist[-3]+" \\\\"
    stringlist[-2] = stringlist[-2]+" \\\\"
    stringlist[-1] = stringlist[-1]+" \\\\ \hline"
    prefix = ["\\begin{table}[]", "\\begin{tabular}{lll}"]
    suffix = ["\end{tabular}", "\caption{Model summary for test.}", "\label{tab:model-summary}" , "\end{table}"]
    stringlist = prefix + stringlist + suffix 
    out_str = " \n".join(stringlist)
    out_str = out_str.replace("_", "\_")
    out_str = out_str.replace("#", "\#")
    print(out_str)

def m2texbase(model, name):
    import pandas as pd
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print('\n\nbase\n\n')
    table=pd.DataFrame(columns=["Name","Type","Shape"])
    for i, layer in enumerate(model.layers):
        # if i == 0:
        #     for base_layer in layer.layers:
        #         table = table.append({"Name":base_layer.name, "Type": base_layer.__class__.__name__,"Shape":base_layer.output_shape}, ignore_index=True)
        # else:
        #     table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
        table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
    print(table.to_string())
    print('\n\n\n')
    table.to_csv(f'./models_csv/{name}.csv', encoding='utf-8', index=False)

    
def save_model(self, name):
    self.model.save('./models/' + name)


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
    Timer(0.1, record_train_gpu_load).start()
    train_gpu_load.extend(get_gpu_util())

def record_test_gpu_load():
    if not collecting_test_gpu:
        return
    Timer(0.1, record_test_gpu_load).start()
    test_gpu_load.extend(get_gpu_util())

def record_train_gpu():
    if not collecting_train_gpu:
        return
    Timer(0.1, record_train_gpu).start()
    train_gpu_usage.extend(get_gpu_memory())

def record_test_gpu():
    if not collecting_test_gpu:
        return
    Timer(0.1, record_test_gpu).start()
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
    trainX, trainY, testX, testY = DataUtils().api_load_medical_data(250, 150, percentage_split)

    print("creating model...")
    model = None

    if model_name == 'one_vgg':
        model = define_model_one_VGG_block(learning_rate, dropout_rate)
    elif model_name == 'two_vgg':
        model = define_model_two_VGG_blocks(learning_rate, dropout_rate)
    elif model_name == 'three_vgg':
        model = define_model_three_VGG_blocks(learning_rate, dropout_rate)

    m2texbase( model, model_name)
    
    # print("compiling model...")
    # record_train_gpu()
    # record_train_gpu_load()
    # start_train = time.time()
    # history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=1)
    # global collecting_train_gpu
    # collecting_train_gpu = False
    # train_time = time.time() - start_train

    # print("evaluating model...")
    # record_test_gpu()
    # record_test_gpu_load()
    # start_test = time.time()
    # _, acc = model.evaluate(testX, testY, verbose=0)
    # global collecting_test_gpu
    # collecting_test_gpu = False
    # test_time = time.time() - start_test

    # print("ACCURACY: ", acc)

    # print("\n\n########## history ################")
    # train_loss = history.history['loss']
    # val_loss   = history.history['val_loss']
    # train_acc  = history.history['accuracy']
    # val_acc    = history.history['val_accuracy'] 
    # print("train loss = ", train_loss)
    # print("val loss = ", val_loss)
    # print("train acc = ", train_acc)
    # print("val acc = ", val_acc)
    # print('gpu load', test_gpu_load)
    # print("\n\n")
    

    # diagnostic = Diagnostic()

    # diagnostic.model = model_name
    # diagnostic.activation_func = activation_func
    # diagnostic.batch_size = batch_size
    # diagnostic.dropout_rate = dropout_rate
    # diagnostic.epochs = epochs
    # diagnostic.learning_rate = learning_rate
    # diagnostic.momentum = momentum
    # diagnostic.optimizer = optimizer
    # diagnostic.train_time = train_time
    # diagnostic.test_time = test_time
    # diagnostic.test_acc = acc
    # diagnostic.train_samples = len(trainX)
    # diagnostic.test_samples = len(testX)
    # diagnostic.percentage = percentage_split + 0.05

    # gpu_train_diagnostics = []
    # gpu_test_diagnostics = []
    # fit_histories = []

    # for count, (gpu, load) in enumerate(zip(train_gpu_usage, train_gpu_load)):
    #     gpu_train = GPUUsageTrain()
    #     gpu_train.time = count/10
    #     gpu_train.usage = gpu
    #     gpu_train.load = load
    #     gpu_train_diagnostics.append(gpu_train)
    
    # for count, (gpu, load) in enumerate(zip(test_gpu_usage, test_gpu_load)):
    #     gpu_test = GPUUsageTest()
    #     gpu_test.time = count/10
    #     gpu_test.usage = gpu
    #     gpu_test.load = load
    #     gpu_test_diagnostics.append(gpu_test)
    
    # for count, (tl, vl, ta, va) in enumerate(zip(train_loss, val_loss, train_acc, val_acc)):
    #     hist = FitHistory()
    #     hist.epoch = count + 1
    #     hist.train_loss = tl
    #     hist.val_loss = vl
    #     hist.train_acc = ta
    #     hist.val_acc = va
    #     fit_histories.append(hist)
    
    # diagnostic.history = fit_histories
    # diagnostic.gpu_test = gpu_test_diagnostics
    # diagnostic.gpu_train = gpu_train_diagnostics

    # save_diagnostics(diagnostic)

if __name__ == '__main__':
    main()


# epoch [10, 30, 60, 100]
# 