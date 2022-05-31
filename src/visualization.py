from random import random
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
from keras import backend as K

from keras.models import Model

import tensorflow as tf

from data_utils import DataUtils

import subprocess as sp
import os
from threading import Thread , Timer
import sched, time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore



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

        self.model = models.Sequential()

        for l in self.base_model.layers:
            self.model.add(l)

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
    
    def get_model(self):
        return self.model

def get_heatmap(gradcam, model, index, cat_index):
    cam = gradcam(CategoricalScore(cat_index), model.trainX[index])

    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)

    return heatmap

def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def plot_embedding(x, y):
    cm = plt.cm.get_cmap('RdYlGn')
    f = plt.figure(figsize=(13, 13))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=y, cmap=cm)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    
    plt.show()
    plt.close()


def gradcam(model, alpha=0.5):
    gradcam = GradcamPlusPlus(model.get_model(), model_modifier=ReplaceToLinear(), clone=True)

    test_n_start = 202
    test_p_start = 404

    train_n_start = 1200
    train_p_start = 2200
    for i in range(100):
        covh = get_heatmap(gradcam, model, 0 + i, 0)
        norh = get_heatmap(gradcam, model, test_n_start + i, 1)
        pneh = get_heatmap(gradcam, model, test_p_start + i, 2)

        plt.subplot(131)
        plt.title('Covid')
        plt.imshow(model.trainX[0 + i])
        plt.imshow(covh, cmap='jet', alpha=alpha)

        plt.subplot(132)
        plt.title('Normal')
        plt.imshow(model.trainX[1200 + i])
        plt.imshow(norh, cmap='jet', alpha=alpha)

        plt.subplot(133)
        plt.title('Pneu')
        plt.imshow(model.trainX[2200 + i])
        plt.imshow(pneh, cmap='jet', alpha=alpha)

        plt.tight_layout()
        plt.savefig(f'images/rad/gradcam/gradcam{i}_alpha{alpha}.png')
        plt.close()


def score_function(output):
    return (output[0][1], output[1][1], output[2][1])


def gradcam_plus_plus_cust(model, alpha=0.5):
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    replace2linear = ReplaceToLinear()

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model.get_model(),
                            model_modifier=replace2linear,
                            clone=True)

    for img in range(100):
        
        X = np.asanyarray([model.trainX[1 + img], model.trainX[1200 + img], model.trainX[2200 + img]])

        images = [model.trainX[1 + img], model.trainX[1200 + img], model.trainX[2200 + img]]

        

        # Generate heatmap with GradCAM++
        cam = gradcam(score_function,
                    X,
                    penultimate_layer=-1)
        

        ## Since v0.6.0, calling `normalize()` is NOT necessary.
        # cam = normalize(cam)
        image_titles = ['covid', 'normal', 'viral']
        # Render
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i, title in enumerate(image_titles):
            heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
            ax[i].set_title(title, fontsize=16)
            ax[i].imshow(images[i])
            ax[i].imshow(heatmap, cmap='jet', alpha=alpha)
            ax[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'images/rad/grad_cam_plus_plus/gradcam_plus_plus{img}_alpha{alpha}.png')
        plt.close()
    # plt.show()


def score_cam(model, alpha=0.5):
    from tf_keras_vis.scorecam import Scorecam

    # Create ScoreCAM object
    scorecam = Scorecam(model.get_model())

    for i in range(100):
        X = np.asanyarray([model.trainX[1  + i], model.trainX[1200 + i], model.trainX[2200 + i]])
        images = [model.trainX[1 + i], model.trainX[1200 + i], model.trainX[2200 + i]]
        image_titles = ['covid', 'normal', 'viral']

        # Generate heatmap with ScoreCAM
        cam = scorecam(score_function, X, penultimate_layer=-1)

        ## Since v0.6.0, calling `normalize()` is NOT necessary.
        # cam = normalize(cam)

        # Render
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for j, title in enumerate(image_titles):
            heatmap = np.uint8(cm.jet(cam[j])[..., :3] * 255)
            ax[j].set_title(title, fontsize=16)
            ax[j].imshow(images[j])
            ax[j].imshow(heatmap, cmap='jet', alpha=alpha)
            ax[j].axis('off')
        plt.tight_layout()
        plt.savefig(f'images/rad/score_cam/score_cam{i}_alpha{alpha}.png')
        plt.close()
    # plt.show()


def vanilla_saliency(model):
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    replace2linear = ReplaceToLinear()
    from tensorflow.keras import backend as K
    from tf_keras_vis.saliency import Saliency
    # from tf_keras_vis.utils import normalize

    # score = CategoricalScore([1, 2, 3])

    # Create Saliency object.
    saliency = Saliency(model.get_model(),
                        model_modifier=replace2linear,
                        clone=True)

    for k in range(100):

        
        
        X = np.asanyarray([model.trainX[1 + k], model.trainX[1200 + k], model.trainX[2200 + k]])
        image_titles = ['covid', 'normal', 'viral']

        # Generate saliency map
        saliency_map = saliency(score_function, X)

        # Render
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i, title in enumerate(image_titles):
            ax[i].set_title(title, fontsize=16)
            ax[i].imshow(saliency_map[i], cmap='jet')
            ax[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'images/rad/vanilla_saliency/van{k}.png')
        plt.close()


def smooth_grad(model, noise=0.20):
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    replace2linear = ReplaceToLinear()
    from tensorflow.keras import backend as K
    from tf_keras_vis.saliency import Saliency

    # Create Saliency object.
    saliency = Saliency(model.get_model(),
                        model_modifier=replace2linear,
                        clone=True)

    for i in range(100):
        
        # from tf_keras_vis.utils import normalize

        # score = CategoricalScore([1, 2, 3])

        
        X = np.asanyarray([model.trainX[1 + i], model.trainX[1200 + i], model.trainX[2200 + i]])
        image_titles = ['covid', 'normal', 'viral']

        saliency_map = saliency(score_function,
                            X,
                            smooth_samples=20, # The number of calculating gradients iterations.
                            smooth_noise=noise) # noise spread level.

        ## Since v0.6.0, calling `normalize()` is NOT necessary.
        # saliency_map = normalize(saliency_map)

        # Render
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for j, title in enumerate(image_titles):
            ax[j].set_title(title, fontsize=14)
            ax[j].imshow(saliency_map[j], cmap='jet')
            ax[j].axis('off')
        plt.tight_layout()
        plt.savefig(f'images/rad/smooth_grad/smoothgrad{i}_noise{noise}.png')
        plt.close()
    # plt.show()


def tsne_plot(model, perplexity=30):
    from sklearn.manifold import TSNE

    m = model.get_model()

    print('predicting on new model')
    dense2_layer_model = Model(inputs=m.input , outputs=m.get_layer('dense_2').output)
    out = dense2_layer_model.predict(model.trainX)

    color = model.trainY
    color = [np.argmax(i) for i in color]
    color = np.stack(color, axis=0)

    n_neighbors = 3
    n_components = 2

    X_train = model.trainX
    np.random.shuffle(X_train)
    img_size = 100

    fig = plt.figure(figsize=(12, 12))		#  Specify the width and height of the image 

    # t-SNE Dimensionality reduction and visualization of the final results 
    # ts = TSNE(n_components=n_components, init='pca', random_state=0)
    ts = TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
    
    #  Training models 
    print('reshaping')
    out = out.reshape((3003, out.shape[1]))
    y = ts.fit_transform(out)
    ax1 = fig.add_subplot(3, 1, 2)

    print('plotting raw...')
    cm = 'winter_r' # Adjust the color 
    #cm = colormap()
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    ax1.set_title('t-SNE Scatter Plot', fontsize=14)

    # #  draw S Type curve 3D Images 
    # ax = fig.add_subplot(313, projection='3d')		#  Create subgraphs 
    # ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=color, cmap=cm)		#  Draw a scatter plot , Give different colors to the points of different labels 
    # ax.set_title('Original S-Curve', fontsize=14)
    # ax.view_init(4, -72)		#  Initialize perspective 

    # 

    # t-SNE Dimensionality reduction and visualization of the original image 
    print('performing TSNE...')
    # ts = TSNE(n_components=n_components, init='pca', random_state=0)
    ts = TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
    #  Training models 
    X_test = model.trainX.reshape((len(model.trainX), img_size*img_size*3))
    print(X_test)
    y = ts.fit_transform(X_test)
    for outer in y:
        for inner in outer:
            ran = random()
            inner = inner + (inner*ran/5)
            
    ax1 = fig.add_subplot(3, 1, 1)

    print('plotting transformed...')
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    ax1.set_title('Raw Data Scatter Plot', fontsize=14)

    #  Display images 
    # plt.show()
    print('saving plot...')
    # plt.savefig(f'images/TSNE_RAD_both{perplexity}perp.png')
    plt.show()
    plt.close()


def tsne2_plot(model, perplexity=30):
    from sklearn.manifold import TSNE

    m = model.get_model()

    print('predicting on new model')
    dense2_layer_model = Model(inputs=m.input , outputs=m.get_layer('dense_2').output)
    out = dense2_layer_model.predict(model.trainX)

    color = model.trainY
    color = [np.argmax(i) for i in color]
    color = np.stack(color, axis=0)

    n_neighbors = 3
    n_components = 3

    X_train = model.trainX
    np.random.shuffle(X_train)
    img_size = 100

    fig = plt.figure(figsize=(12, 12))		#  Specify the width and height of the image 

    # t-SNE Dimensionality reduction and visualization of the final results 
    # ts = TSNE(n_components=n_components, init='pca', random_state=0)
    ts = TSNE(n_components=n_components, init='pca', random_state=0, perplexity=perplexity)
    #  Training models 
    print('reshaping')
    out = out.reshape((3003, out.shape[1]))
    y = ts.fit_transform(out)
    ax1 = fig.add_subplot(4, 1, 2)

    print('plotting raw...')
    cm = 'winter_r' # Adjust the color 
    #cm = colormap()
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    ax1.set_title('t-SNE Scatter Plot', fontsize=14)

    #  draw S Type curve 3D Images 
    ax = fig.add_subplot(413, projection='3d')		#  Create subgraphs 
    ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=color, cmap=cm)		#  Draw a scatter plot , Give different colors to the points of different labels 
    ax.set_title('Dense layer ouput', fontsize=14)
    ax.view_init(4, -72)		#  Initialize perspective 

    

    # t-SNE Dimensionality reduction and visualization of the original image 
    print('performing TSNE...')
    # ts = TSNE(n_components=n_components, init='pca', random_state=0)
    ts = TSNE(n_components=n_components, init='pca', random_state=0, perplexity=perplexity)
    #  Training models 
    X_test = model.trainX.reshape((len(model.trainX), img_size*img_size*3))
    y = ts.fit_transform(X_test)
    ax1 = fig.add_subplot(4, 1, 1)

    print('plotting transformed...')
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    ax1.set_title('Raw Data Scatter Plot', fontsize=14)

    ax = fig.add_subplot(414, projection='3d')		#  Create subgraphs 
    ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=color, cmap=cm)		#  Draw a scatter plot , Give different colors to the points of different labels 
    ax.set_title('Raw data', fontsize=14)
    ax.view_init(4, -72)		#  Initialize perspective 


    #  Display images 
    # plt.show()
    print('saving plot...')
    plt.savefig(f'images/TSNE_RAD_both_perplexity_FINAL.png')
    plt.show()
    plt.close()


def plot_actual_predicted(images, pred_classes, class_names):
  fig, axes = plt.subplots(1, 4, figsize=(16, 15))
  axes = axes.flatten()
  
  # plot
  ax = axes[0]
  dummy_array = np.array([[[0, 0, 0, 0]]], dtype='uint8')
  ax.set_title("Base reference")
  ax.set_axis_off()
  ax.imshow(dummy_array, interpolation='nearest')  # plot image
  for k,v in images.items():
    ax = axes[k+1]
    ax.imshow(v, cmap=plt.cm.binary)
    ax.set_title(f"True: %s \nPredict: %s" % (class_names[k], class_names[pred_classes[k]]))
    ax.set_axis_off()  

  plt.tight_layout()
  plt.imshow()


def shap_plot(model: TransferLearningModel, i = 0):
    import shap
    import ssl

    # class label list
    class_names = ['Covid', 'Normal', 'Viral']# example image for each class
    images_dict = dict()
    # for i, l in enumerate(model.trainY):
    #     if len(images_dict)==3:
    #         break
    #     if l not in images_dict.keys():
    #         images_dict[l] = model.trainX[i].reshape((100, 100,3))
    #     images_dict = dict(sorted(images_dict.items()))
        
    images_dict = {0: model.trainX[0 + i].reshape((100, 100, 3)), 1: model.trainX[1200 + i].reshape((100, 100, 3)), 2: model.trainX[2200 + i].reshape((100, 100, 3))}
    images_dict = dict(sorted(images_dict.items()))
    # example image for each class for test set
    x_test_dict = dict()
    # for i, l in enumerate(model.testY):
    #     if len(x_test_dict)==3:
    #         break
    #     if l not in x_test_dict.keys():
    #         x_test_dict[l] = model.testX[i]# order by class
    
    x_test_dict = {0: model.trainX[0 + i].reshape((100, 100, 3)), 1: model.trainX[1200 + i].reshape((100, 100, 3)), 2: model.trainX[2200 + i].reshape((100, 100, 3))}

    x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
    x_test_each_class = np.asarray(x_test_each_class)# Compute predictions
    predictions = model.get_model().predict(x_test_each_class)
    predicted_class = np.argmax(predictions, axis=1)

    # select backgroud for shap
    background = model.trainX[np.random.choice(model.trainX.shape[0], 20, replace=False)]# DeepExplainer to explain predictions of the model
    explainer = shap.DeepExplainer(model.get_model(), background)# compute shap values
    shap_values = explainer.shap_values(x_test_each_class)

    # plot_actual_predicted(images_dict, predicted_class, class_names)
    print()
    shap.image_plot(shap_values, x_test_each_class * 255, show=False)
    plt.savefig(f'img/rad/shap/shap{i}.png')


def lime(model: TransferLearningModel):
    import lime
    from lime import lime_image
    explainer = lime_image.LimeImageExplainer()
    from skimage.segmentation import mark_boundaries

    for folder in ['normal', 'viral']:
        for i in range(100):
            if folder == 'normal':
                i = i+1200
            if folder == 'viral':
                i = i+2200
            explanation = explainer.explain_instance(model.trainX[0 + i].astype('double'), model.get_model().predict, top_labels=5, hide_color=0, num_samples=1000)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.savefig(f'images/rad/lime_{folder}/hide_rest_false{i}')
            plt.close()

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.savefig(f'images/rad/lime_{folder}/hide_rest_true{i}')
            plt.close()  

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.savefig(f'images/rad/lime_{folder}/hide_rest_false_pos_only_false{i}')
            plt.close()

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
            plt.savefig(f'images/rad/lime_{folder}/hide_rest_false_pos_only_false_min_weight{i}')
            plt.close()

            #Select the same class explained on the figures above.
            ind =  explanation.top_labels[0]

            #Map each explanation weight to the corresponding superpixel
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

            #Plot. The visualization makes more sense if a symmetrical colorbar is used.
            plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
            plt.colorbar()
            plt.savefig(f'images/rad/lime/heat_map{i}')
            plt.close()



def main():
    model_name = 'VGG16' # Model                     ***
    batch_size = 32 # batch size           ***
    optimizer = 'adam' # optimizer             
    learning_rate = 0.0001 # learning rate     ***
    epochs = 50 # epochs                   ***
    momentum = 0 # momentum               ***
    activation_func = 'relu' #                 ***
    dropout_rate = 0.0 #                    ***
    index = int(sys.argv[1]) 

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])

    print("loading and preprocessing data...")
    model = TransferLearningModel()
    model.load_data(100, 100)

    print("loading pre-trained model and adding fully connected layers...")
    model.load_pre_trained_model(name=model_name)
    model.add_fully_connected_layers(dense_activation=activation_func, dropout_rate=dropout_rate)
    
    history = model.compile(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)

    shap_plot(model, index)

    # T-SNE
    
    # from sklearn.manifold import TSNE

    # lime(model)

    # tsne_plot(model)
    # tsne2_plot(model)

    # shap_plot(model)
    
    # gradcam_plus_plus_cust(model, alpha=0.3)
    # smooth_grad(model, noise=0.3)
    # score_cam(model, alpha=0.3)
    # gradcam(model, alpha=0.3)
    # vanilla_saliency(model)
    # tsne_plot(model)


    # TSNE
    # print('predicting on new model')
    # dense2_layer_model = Model(inputs=m.input , outputs=m.get_layer('dense_2').output)
    # out = dense2_layer_model.predict(model.trainX)

    # color = model.trainY
    # color = [np.argmax(i) for i in color]
    # color = np.stack(color, axis=0)

    # n_neighbors = 3
    # n_components = 3

    # X_train = model.trainX
    # np.random.shuffle(X_train)
    # img_size = 100

    # fig = plt.figure(figsize=(12, 12))		#  Specify the width and height of the image 

    # # t-SNE Dimensionality reduction and visualization of the final results 
    # ts = TSNE(n_components=n_components, init='pca', random_state=0)
    # #  Training models 
    # print('reshaping')
    # out = out.reshape((3003, out.shape[1]))
    # y = ts.fit_transform(out)
    # ax1 = fig.add_subplot(3, 1, 2)

    # print('plotting raw...')
    # cm = 'winter_r' # Adjust the color 
    # #cm = colormap()
    # plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    # ax1.set_title('t-SNE Scatter Plot', fontsize=14)

    # #  draw S Type curve 3D Images 
    # ax = fig.add_subplot(313, projection='3d')		#  Create subgraphs 
    # ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=color, cmap=cm)		#  Draw a scatter plot , Give different colors to the points of different labels 
    # ax.set_title('Original S-Curve', fontsize=14)
    # ax.view_init(4, -72)		#  Initialize perspective 

    # # t-SNE Dimensionality reduction and visualization of the original image 
    # print('performing TSNE...')
    # ts = TSNE(n_components=n_components, init='pca', random_state=0)
    # #  Training models 
    # X_test = X_train.reshape((len(X_train), img_size*img_size*3))
    # y = ts.fit_transform(X_test)
    # ax1 = fig.add_subplot(3, 1, 1)

    # print('plotting transformed...')
    # plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
    # ax1.set_title('Raw Data Scatter Plot', fontsize=14)

    # #  Display images 
    # # plt.show()
    # print('saving plot...')
    # plt.savefig(f'./TEST_VIS_YOYO100.png')

    # TSNE END



    # weights = m.get_layer('dense_2').get_weights()
    # tsne = TSNE(n_components=2, verbose=1)
    # transformed_weights = tsne.fit_transform(weights)

    # plot_embedding(transformed_weights, model.trainY)


    # model2 = tf.keras.Model(inputs=m.input, outputs=m.layers[-2].output)
    # test_ds = np.concatenate(list(model.trainX))
    # features = model2(test_ds)
    # labels = np.argmax(m(test_ds), axis=-1)
    # tsne = TSNE(n_components=2).fit_transform(features)

    # tx = tsne[:, 0]
    # ty = tsne[:, 1]

    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)

    # colors = ['red', 'blue', 'green']
    # classes = [0, 1, 2]
    # print(classes)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for idx, c in enumerate(colors):
    #     indices = [i for i, l in enumerate(labels) if idx == l]
    #     current_tx = np.take(tx, indices)
    #     current_ty = np.take(ty, indices)
    #     ax.scatter(current_tx, current_ty, c=c, label=classes[idx])

    # ax.legend(loc='best')
    # plt.show()

    # GRADCAM 
    # gradcam = GradcamPlusPlus(model.get_model(), model_modifier=ReplaceToLinear(), clone=True)

    # test_n_start = 202
    # test_p_start = 404

    # train_n_start = 1200
    # train_p_start = 2200
    # for i in range(10):
    #     covh = get_heatmap(gradcam, model, 0 + i, 0)
    #     norh = get_heatmap(gradcam, model, test_n_start + i, 1)
    #     pneh = get_heatmap(gradcam, model, test_p_start + i, 2)

    #     plt.subplot(131)
    #     plt.title('Covid')
    #     plt.imshow(model.trainX[0])
    #     plt.imshow(covh, cmap='jet', alpha=0.5)

    #     plt.subplot(132)
    #     plt.title('Normal')
    #     plt.imshow(model.trainX[1200])
    #     plt.imshow(norh, cmap='jet', alpha=0.5)

    #     plt.subplot(133)
    #     plt.title('Pneu')
    #     plt.imshow(model.trainX[2200])
    #     plt.imshow(pneh, cmap='jet', alpha=0.5)
    #     plt.colorbar()

    #     plt.tight_layout()
    #     plt.savefig(f'./gradcams/{i}TEST_VIS.png')
    #     plt.show()
    #     plt.close()



    # _, acc = model.model.evaluate(model.testX, model.testY, verbose=0)
    # print("ACCURACY: ", acc)



if __name__ == '__main__':
    main()


# epoch [10, 30, 60, 100]
# 