from visualization import TransferLearningModel
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.scorecam import ScoreCAM

test_n_start = 202
test_p_start = 404

train_n_start = 1200
train_p_start = 2200

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

def score_cam(model, alpha=0.5):
    gradcam = ScoreCAM(model.get_model(), model_modifier=ReplaceToLinear(), clone=True)

    print('COV')
    for i in range(500):
        h = get_heatmap_train(gradcam, model, i, 0)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print('NORM')
    for i in range(500):
        h = get_heatmap_train(gradcam, model, i + train_n_start, 1)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print('PNEU')
    for i in range(500):
        h = get_heatmap_train(gradcam, model, i + train_p_start, 1)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    print('HALF')
    for i in range(100):
        h = get_heatmap_test(gradcam, model, i, 0)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(100):
        h = get_heatmap_test(gradcam, model, i + test_n_start, 1)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(100):
        h = get_heatmap_test(gradcam, model, i + test_p_start, 1)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/scorecam/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()

def gradcam(model, alpha=0.5):
    gradcam = GradcamPlusPlus(model.get_model(), model_modifier=ReplaceToLinear(), clone=True)

    for i in range(500):
        h = get_heatmap_train(gradcam, model, i, 0)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcam/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(500):
        h = get_heatmap_train(gradcam, model, i + train_n_start, 1)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcam/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(500):
        h = get_heatmap_train(gradcam, model, i + train_p_start, 1)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcam/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(100):
        h = get_heatmap_test(gradcam, model, i, 0)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcam/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(100):
        h = get_heatmap_test(gradcam, model, i + test_n_start, 1)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcam/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(100):
        h = get_heatmap_test(gradcam, model, i + test_p_start, 1)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcam/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()


def score_function_cov(output):
    return (output[0][1])

def score_function_norm(output):
    return (output[0][1])

def score_function_viral(output):
    return (output[0][1])


def layer_cam(model, alpha=0.5):
    replace2linear = ReplaceToLinear()
    i = 1
    gradcam = Layercam(model.get_model(), model_modifier=replace2linear, clone=True)

    h = gradcam(score_function_cov,
                    model.trainX[i],
                    penultimate_layer=-1)
    h = np.uint8(cm.jet(h[0])[..., :3] * 255)
    #plt.imshow(model.trainX[i])
    plt.imshow(h, cmap='jet', alpha=alpha)
    # plt.savefig(f'img/rad/layercam/COVID{i}_alpha{alpha}TRAIN.png')
    plt.show()
    #plt.close()


def gradcam_plus_plus(model, alpha=0.5):
    replace2linear = ReplaceToLinear()

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model.get_model(),
                            model_modifier=replace2linear,
                            clone=True)
    
    for i in range(500):
        h = gradcam(score_function_cov,
                    model.trainX[i],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.trainX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcamplusplus/COVID{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(500):
        h = gradcam(score_function_norm,
                    model.trainX[i + train_n_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.trainX[i + train_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcamplusplus/NORMAL{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(500):
        h = gradcam(score_function_viral,
                    model.trainX[i + train_p_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.trainX[i + train_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcamplusplus/PNEU{i}_alpha{alpha}TRAIN.png')
        plt.close()
    
    for i in range(100):
        h = gradcam(score_function_cov,
                    model.testX[i],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.testX[i])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcamplusplus/COVID{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(100):
        h = gradcam(score_function_norm,
                    model.testX[i + test_n_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.testX[i + test_n_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcamplusplus/NORMAL{i}_alpha{alpha}TEST.png')
        plt.close()
    
    for i in range(100):
        h = gradcam(score_function_viral,
                    model.testX[i + test_p_start],
                    penultimate_layer=-1)
        h = np.uint8(cm.jet(h[0])[..., :3] * 255)
        plt.imshow(model.testX[i + test_p_start])
        plt.imshow(h, cmap='jet', alpha=alpha)
        plt.savefig(f'img/rad/gradcamplusplus/PNEU{i}_alpha{alpha}TEST.png')
        plt.close()


def vanilla_saliency(model):
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    replace2linear = ReplaceToLinear()
    from tensorflow.keras import backend as K
    from tf_keras_vis.saliency import Saliency

    saliency = Saliency(model.get_model(),
                        model_modifier=replace2linear,
                        clone=True)
    
    for i in range(500):
        X = np.asanyarray([model.trainX[i]])
        saliency_map = saliency(score_function_cov, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/van_saliency/COVID{i}TRAIN.png')
        plt.close()
    
    for i in range(500):
        X = np.asanyarray([model.trainX[i + train_n_start]])
        saliency_map = saliency(score_function_norm, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/van_saliency/NORM{i}TRAIN.png')
        plt.close()
    
    for i in range(500):
        X = np.asanyarray([model.trainX[i + train_p_start]])
        saliency_map = saliency(score_function_viral, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/van_saliency/VIRAL{i}TRAIN.png')
        plt.close()
    
    for i in range(100):
        X = np.asanyarray([model.testX[i]])
        saliency_map = saliency(score_function_cov, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/van_saliency/COVID{i}TEST.png')
        plt.close()
    
    for i in range(100):
        X = np.asanyarray([model.testX[i + test_n_start]])
        saliency_map = saliency(score_function_norm, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/van_saliency/NORM{i}TEST.png')
        plt.close()
    
    for i in range(100):
        X = np.asanyarray([model.testX[i + test_p_start]])
        saliency_map = saliency(score_function_viral, X)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/van_saliency/VIRAL{i}TEST.png')
        plt.close()


def smooth_grad(model, noise=0.20):
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    replace2linear = ReplaceToLinear()
    from tensorflow.keras import backend as K
    from tf_keras_vis.saliency import Saliency

    saliency = Saliency(model.get_model(),
                        model_modifier=replace2linear,
                        clone=True)
    
    for i in range(500):
        X = np.asanyarray([model.trainX[i]])
        saliency_map = saliency(score_function_cov, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/smoothgrad/COVID{i}TRAIN{noise}.png')
        plt.close()
    
    for i in range(500):
        X = np.asanyarray([model.trainX[i + train_n_start]])
        saliency_map = saliency(score_function_norm, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/smoothgrad/NORM{i}TRAIN{noise}.png')
        plt.close()
    
    for i in range(500):
        X = np.asanyarray([model.trainX[i + train_p_start]])
        saliency_map = saliency(score_function_viral, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/smoothgrad/VIRAL{i}TRAIN{noise}.png')
        plt.close()
    
    for i in range(100):
        X = np.asanyarray([model.testX[i]])
        saliency_map = saliency(score_function_cov, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/smoothgrad/COVID{i}TEST{noise}.png')
        plt.close()
    
    for i in range(100):
        X = np.asanyarray([model.testX[i + test_n_start]])
        saliency_map = saliency(score_function_norm, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/smoothgrad/NORM{i}TEST{noise}.png')
        plt.close()
    
    for i in range(100):
        X = np.asanyarray([model.testX[i + test_p_start]])
        saliency_map = saliency(score_function_viral, X, smooth_samples=20, smooth_noise=noise)
        plt.imshow(saliency_map[0], cmap='jet')
        plt.savefig(f'img/rad/smoothgrad/VIRAL{i}TEST{noise}.png')
        plt.close()


def save_norm(model, alpha=0.5):
    replace2linear = ReplaceToLinear()

    
    for i in range(500):
        plt.imshow(model.trainX[i])
        plt.savefig(f'img/rad/gradcamplusplus/COVID{i}_alpha{alpha}TRAIN_BASE.png')
        plt.close()
    
    for i in range(500):
        plt.imshow(model.trainX[i + train_n_start])
        plt.savefig(f'img/rad/gradcamplusplus/NORMAL{i}_alpha{alpha}TRAIN_BASE.png')
        plt.close()
    
    for i in range(500):
        plt.imshow(model.trainX[i + train_p_start])
        plt.savefig(f'img/rad/gradcamplusplus/PNEU{i}_alpha{alpha}TRAIN_BASE.png')
        plt.close()
    
    for i in range(100):
        plt.imshow(model.testX[i])
        plt.savefig(f'img/rad/gradcamplusplus/COVID{i}_alpha{alpha}TEST_BASE.png')
        plt.close()
    
    for i in range(100):
        plt.imshow(model.testX[i + test_n_start])
        plt.savefig(f'img/rad/gradcamplusplus/NORMAL{i}_alpha{alpha}TEST_BASE.png')
        plt.close()
    
    for i in range(100):
        plt.imshow(model.testX[i + test_p_start])
        plt.savefig(f'img/rad/gradcamplusplus/PNEU{i}_alpha{alpha}TEST_BASE.png')
        plt.close()


def tsne_simple(model: TransferLearningModel, perplexity=30, components=2):
    hidden_features = model.get_model().predict(model.trainX)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(hidden_features)

    tsne = TSNE(n_components=components, verbose = 1, perplexity=perplexity)
    tsne_results = tsne.fit_transform(pca_result)

    from keras.utils import np_utils

    class_names = ['Covid-19', 'Normal', 'Viral Pneumonia']
    y_test_cat = np_utils.to_categorical(model.trainY, num_classes = 3)
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(10,10))
    for cl in range(3):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=class_names[cl])
    plt.legend()
    plt.savefig(f'img/rad/tsne{perplexity}.png')


def tsne_simple_base(model: TransferLearningModel, perplexity=30, components=2):

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    img_size = 100
    pca = PCA(n_components=3)
    X_test = model.trainX.reshape((len(model.trainX), img_size*img_size*3))
    pca_result = pca.fit_transform(X_test)

    tsne = TSNE(n_components=components, verbose = 1, perplexity=perplexity)
    tsne_results = tsne.fit_transform(pca_result)

    from keras.utils import np_utils

    class_names = ['Covid-19', 'Normal', 'Viral Pneumonia']
    y_test_cat = np_utils.to_categorical(model.trainY, num_classes = 3)
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(10,10))
    for cl in range(3):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=class_names[cl])
    plt.legend()
    plt.savefig(f'img/rad/tsne{perplexity}BASE.png')


def main():
    model_name = 'VGG16' # Model                     ***
    batch_size = 32 # batch size           ***
    optimizer = 'adam' # optimizer             
    learning_rate = 0.0001 # learning rate     ***
    epochs = 100 # epochs                   ***
    momentum = 0 # momentum               ***
    activation_func = 'relu' #                 ***
    dropout_rate = 0.0

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])

    model = TransferLearningModel()
    model.load_data(100, 100)

    model.load_pre_trained_model(name=model_name)
    model.add_fully_connected_layers(dense_activation=activation_func, dropout_rate=dropout_rate)
    
    history = model.compile(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum, optimizer=optimizer)

    score_cam(model, 0.3)
    # layer_cam(model)
    # tsne_simple(model)
    # tsne_simple_base(model)

    # save_norm(model)

    # gradcam(model, 0.3)
    # gradcam(model, 0.5)
    # # gradcam(model, 0.8)
    # gradcam_plus_plus(model)
    # gradcam_plus_plus(model, 0.3)
    # # gradcam_plus_plus(model, 0.8)
    # vanilla_saliency(model)
    # smooth_grad(model)
    # smooth_grad(model, 0.1)
    # smooth_grad(model, 0.3)

if __name__ == '__main__':
    main()