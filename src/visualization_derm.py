from derm_percentage_test import TransferLearningModel
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

test_n_start = 16
test_p_start = 32

train_n_start = 114
train_p_start = 491

def get_heatmap_train(gradcam, model, index, cat_index):
    cam = gradcam(CategoricalScore(cat_index), model.trainX[index])

    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)

    return heatmap

def get_heatmap_test(gradcam, model, index, cat_index):
    cam = gradcam(CategoricalScore(cat_index), model.testX[index])

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

    for i in range(10):
        h = get_heatmap_train(gradcam, model, i, 0)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_train(gradcam, model, i + train_n_start, 1)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_train(gradcam, model, i + train_p_start, 1)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i, 0)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i + test_n_start, 1)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i + test_p_start, 1)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()

def score_cam_2(model, alpha=0.5):
    replace2linear = ReplaceToLinear()
    from tf_keras_vis.scorecam import ScoreCAM

    # Create GradCAM++ object
    gradcam = ScoreCAM(model.get_model())


    print("first")
    for i in range(10):
        score = CategoricalScore([0])
        #h = get_heatmap_train(gradcam, model, i, 0)
        X = model.trainX[i]
        cam = gradcam(score, X, penultimate_layer=-1)
        h = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/scorecam/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print("second")
    for i in range(10):
        h = get_heatmap_train(gradcam, model, i + train_n_start, 1)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_train(gradcam, model, i + train_p_start, 1)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i, 0)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i + test_n_start, 1)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i + test_p_start, 1)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcam/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()
    


def score_cam(model, alpha=0.5):
    from tf_keras_vis.scorecam import ScoreCAM

    gradcam = ScoreCAM(model.get_model(), model_modifier=ReplaceToLinear(), clone=True)



    print('COV')
    for i in range(10):
        score = []
        h = get_heatmap_train(gradcam, model, i, 0)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print('NORM')
    for i in range(10):
        h = get_heatmap_train(gradcam, model, i + train_n_start, 1)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print('PNEU')
    for i in range(10):
        h = get_heatmap_train(gradcam, model, i + train_p_start, 1)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print('HALF')
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i, 0)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i + test_n_start, 1)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = get_heatmap_test(gradcam, model, i + test_p_start, 1)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()


def base_images(model):
    for i in range(10):
        plt.imshow(model.trainX[i])
        plt.savefig(f'img/derm/base/COVID{i}TRAIN.png')
        plt.close()
    
    for i in range(10):
        plt.imshow(model.trainX[i + train_n_start])
        plt.savefig(f'img/derm/base/NORMAL{i}TRAIN.png')
        plt.close()
    
    for i in range(10):
        plt.imshow(model.trainX[i + train_p_start])
        plt.savefig(f'img/derm/base/PNEU{i}TRAIN.png')
        plt.close()
    
    for i in range(10):
        plt.imshow(model.testX[i])
        plt.savefig(f'img/derm/base/COVID{i}TEST.png')
        plt.close()
    
    for i in range(10):
        plt.imshow(model.testX[i + test_n_start])
        plt.savefig(f'img/derm/base/NORMAL{i}TEST.png')
        plt.close()
    
    for i in range(10):
        plt.imshow(model.testX[i + test_p_start])
        plt.savefig(f'img/derm/base/PNEU{i}TEST.png')
        plt.close()


def score_function_cov(output):
    return (output[0][1])

def score_function_norm(output):
    return (output[0][1])

def score_function_viral(output):
    return (output[0][1])


def gradcam_plus_plus(model, alpha=0.5):
    replace2linear = ReplaceToLinear()

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model.get_model(),
                            model_modifier=replace2linear,
                            clone=True)
    
    for i in range(10):
        h = gradcam(score_function_cov,
                    model.trainX[i],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcamplusplus/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = gradcam(score_function_norm,
                    model.trainX[i + train_n_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcamplusplus/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = gradcam(score_function_viral,
                    model.trainX[i + train_p_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcamplusplus/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(10):
        h = gradcam(score_function_cov,
                    model.testX[i],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcamplusplus/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = gradcam(score_function_norm,
                    model.testX[i + test_n_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcamplusplus/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(10):
        h = gradcam(score_function_viral,
                    model.testX[i + test_p_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/derm/gradcamplusplus/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()








def vanilla_saliency(model):
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    replace2linear = ReplaceToLinear()
    from tensorflow.keras import backend as K
    from tf_keras_vis.saliency import Saliency

    saliency = Saliency(model.get_model(),
                        model_modifier=replace2linear,
                        clone=True)
    
    for i in range(10):
        X = np.asanyarray([model.trainX[i]])
        saliency_map = saliency(score_function_cov, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/van_saliency/COVID{i}TRAIN.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.trainX[i + train_n_start]])
        saliency_map = saliency(score_function_norm, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/van_saliency/NORM{i}TRAIN.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.trainX[i + train_p_start]])
        saliency_map = saliency(score_function_viral, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/van_saliency/VIRAL{i}TRAIN.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.testX[i]])
        saliency_map = saliency(score_function_cov, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/van_saliency/COVID{i}TEST.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.testX[i + test_n_start]])
        saliency_map = saliency(score_function_norm, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/van_saliency/NORM{i}TEST.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.testX[i + test_p_start]])
        saliency_map = saliency(score_function_viral, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/van_saliency/VIRAL{i}TEST.png')
        plt.close()


def smooth_grad(model, noise=0.20):
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    replace2linear = ReplaceToLinear()
    from tensorflow.keras import backend as K
    from tf_keras_vis.saliency import Saliency

    saliency = Saliency(model.get_model(),
                        model_modifier=replace2linear,
                        clone=True)
    
    for i in range(10):
        X = np.asanyarray([model.trainX[i]])
        saliency_map = saliency(score_function_cov, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/smoothgrad/COVID{i}TRAIN{noise}.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.trainX[i + train_n_start]])
        saliency_map = saliency(score_function_norm, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/smoothgrad/NORM{i}TRAIN{noise}.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.trainX[i + train_p_start]])
        saliency_map = saliency(score_function_viral, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/smoothgrad/VIRAL{i}TRAIN{noise}.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.testX[i]])
        saliency_map = saliency(score_function_cov, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/smoothgrad/COVID{i}TEST{noise}.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.testX[i + test_n_start]])
        saliency_map = saliency(score_function_norm, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/smoothgrad/NORM{i}TEST{noise}.png')
        plt.close()
    
    for i in range(10):
        X = np.asanyarray([model.testX[i + test_p_start]])
        saliency_map = saliency(score_function_viral, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/derm/smoothgrad/VIRAL{i}TEST{noise}.png')
        plt.close()





def tsne2_plot(model, perplexity=30):
    from sklearn.manifold import TSNE
    from keras.models import Model

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
    out = out.reshape((2239, out.shape[1]))
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
    plt.savefig(f'images/TSNE_DERM_both_perplexity{perplexity}_FINAL.png')
    # plt.show()
    plt.close()



def plot_actual_predicted(images, pred_classes, class_names):
  fig, axes = plt.subplots(1, 4, figsize=(16, 15))
  axes = axes.flatten()
  
  # plot
  ax = axes[0]
  dummy_array = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype='uint8')
  ax.set_title("Base reference")
  ax.set_axis_off()
  ax.imshow(dummy_array, interpolation='nearest')  # plot image
  for k,v in images.items():
    ax = axes[k+1]
    ax.imshow(v, cmap=plt.cm.binary)
    # ax.set_title(f"True: %s \nPredict: %s" % (class_names[k], class_names[pred_classes[k]]))
    ax.set_axis_off()  

  plt.tight_layout()
  plt.imshow()


def shap_plot(model: TransferLearningModel, i = 0):
    import shap
    import ssl

    # class label list
    class_names = ['Actinic Keratosis', 'Basal cell carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']# example image for each class
    images_dict = dict()
     
    images_dict = {0: model.trainX[0 + 1].reshape((100, 100, 3)), 1: model.trainX[train_n_start +2 + 1].reshape((100, 100, 3)), 2: model.trainX[train_p_start + 2 + 1].reshape((100, 100, 3)),
                3: model.trainX[586 + 1].reshape((100, 100, 3)), 4: model.trainX[1024 + 1].reshape((100, 100, 3)), 5: model.trainX[1381 + 1].reshape((100, 100, 3)),
                6: model.trainX[1843 + 1].reshape((100, 100, 3)), 7: model.trainX[1920 + 1].reshape((100, 100, 3)), 8: model.trainX[2101 + 1].reshape((100, 100, 3))}
    images_dict = dict(sorted(images_dict.items()))
    x_test_dict = dict()
    
    x_test_dict = {0: model.trainX[0 + 1].reshape((100, 100, 3)), 1: model.trainX[train_n_start +2 + 1].reshape((100, 100, 3)), 2: model.trainX[train_p_start + 2 + 1].reshape((100, 100, 3)),
                3: model.trainX[586 + 1].reshape((100, 100, 3)), 4: model.trainX[1024 + 1].reshape((100, 100, 3)), 5: model.trainX[1381 + 1].reshape((100, 100, 3)),
                6: model.trainX[1843 + 1].reshape((100, 100, 3)), 7: model.trainX[1920 + 1].reshape((100, 100, 3)), 8: model.trainX[2101 + 1].reshape((100, 100, 3))}

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
    plt.savefig(f'img/derm/shap/shap{i}.png')
    


def lime(model: TransferLearningModel):
    import lime
    from lime import lime_image
    explainer = lime_image.LimeImageExplainer()
    from skimage.segmentation import mark_boundaries

    for folder in ['Actinic_Keratosis', 'Basal_cell_carcinoma', 'Dermatofibroma']:
        for i in range(10):
            if folder == 'Basal_cell_carcinoma':
                i = i+train_n_start
            if folder == 'Dermatofibroma':
                i = i+train_p_start
            explanation = explainer.explain_instance(model.trainX[0 + i].astype('double'), model.get_model().predict, top_labels=5, hide_color=0, num_samples=2000)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.imshow(mark_boundaries(temp, mask))
            plt.savefig(f'img/report/{folder}hide_rest_false{i}')
            plt.close()

            # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
            # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            # #plt.savefig(f'img/derm/lime/{folder}hide_rest_true{i}')
            # plt.close()  

            # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
            # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            # #plt.savefig(f'img/derm/lime/{folder}hide_rest_false_pos_only_false{i}')
            # plt.close()

            # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
            # #plt.savefig(f'img/derm/lime/{folder}hide_rest_false_pos_only_false_min_weight{i}')
            # plt.close()

            # #Select the same class explained on the figures above.
            # ind =  explanation.top_labels[0]

            # #Map each explanation weight to the corresponding superpixel
            # dict_heatmap = dict(explanation.local_exp[ind])
            # heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

            # #Plot. The visualization makes more sense if a symmetrical colorbar is used.
            # plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
            # plt.colorbar()
            # #plt.savefig(f'img/derm/lime/heat_map{i}.png')
            
            # plt.close()

            plt.imshow(temp)
            plt.savefig(f'img/report/base{folder}{i}.png')
            plt.close()


def tsne_simple(model: TransferLearningModel, perplexity=30, components=2):
    hidden_features = model.get_model().predict(model.trainX)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=9)
    pca_result = pca.fit_transform(hidden_features)

    tsne = TSNE(n_components=components, verbose = 1, perplexity=perplexity)
    tsne_results = tsne.fit_transform(pca_result)

    from keras.utils import np_utils

    class_names = ['Actinic Keratosis', 'Basal cell carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']

    y_test_cat = np_utils.to_categorical(model.trainY, num_classes = 9)
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(10,10))
    for cl in range(9):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=class_names[cl])
    plt.legend()
    plt.savefig(f'img/derm/tsne{perplexity}.png')



def main():
    model_name = 'VGG16'
    batch_size = 32
    optimizer = 'adam'            
    learning_rate = 0.0001
    epochs = 100
    momentum = 0 
    activation_func = 'relu' 
    dropout_rate = 0.0 

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])

    model = TransferLearningModel()
    model.load_data(100, 100, 1)

    model.load_pre_trained_model(name=model_name)
    model.add_fully_connected_layers(dense_activation=activation_func, dropout_rate=dropout_rate)
    
    history = model.compile(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)

    # gradcam(model, 0.3)
    # gradcam(model, 0.5)
    # gradcam(model, 0.8)
    # gradcam_plus_plus(model)
    # gradcam_plus_plus(model, 0.3)
    # gradcam_plus_plus(model, 0.8)
    # vanilla_saliency(model)
    # smooth_grad(model)
    # smooth_grad(model, 0.1)
    # smooth_grad(model, 0.3)
    # tsne2_plot(model, 20)
    # tsne2_plot(model, 60)
    # tsne2_plot(model, 120)
    # tsne2_plot(model, 200)
    # lime(model)
    # shap_plot(model, 6)
    # tsne_simple(model)
    # tsne_simple(model, 210)
    # tsne_simple(model, 220)
    # tsne_simple(model, 140)
    # tsne_simple(model, 150)
    # tsne_simple(model, 160)
    # tsne_simple(model, 170)
    # tsne_simple(model, 180)
    # tsne_simple(model, 190)
    # tsne_simple(model, 1100)
    # tsne_simple(model, 1000)
    # tsne_simple(model, 1200)
    # base_images(model)
    # lime(model)
    score_cam_2(model, 0.3)
    
    # https://becominghuman.ai/visualizing-representations-bd9b62447e38
    
    

if __name__ == '__main__':
    main()