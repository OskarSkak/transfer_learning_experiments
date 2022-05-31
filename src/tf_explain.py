import tensorflow as tf

from tf_explain_local.tf_explain.core import GradCAM

IMAGE_PATH = '../data/COVID-19_Radiography_Dataset/archive/COVID_all/COVID-1.png'

model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)

model.summary()
data = ([img], None)

tabby_cat_class_index = 281
explainer = GradCAM()
# Compute GradCAM on VGG16
grid = explainer.explain(
    data, model, class_index=tabby_cat_class_index, layer_name="block5_conv3"
)
explainer.save(grid, ".", "grad_cam.png")
