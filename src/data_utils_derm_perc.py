from tensorflow.keras.utils import to_categorical
import glob 
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataUtils():
    def api_load_medical_data(self, width=32, height = 32, percentage=1):
        all_test_x = []
        all_test_y = []
        all_train_x = []
        all_train_y = []

        #  smallest shape: 450, 600

        #Actinic_keratosis
        all_actinic_keratosis = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/actinic_keratosis', [], width, height)
        train_actinic_keratosis_x, _ = np.split(all_actinic_keratosis, [int(len(all_actinic_keratosis)*percentage)])

        train_actinic_keratosis_y = np.empty(len(train_actinic_keratosis_x))
        train_actinic_keratosis_y.fill(0)
        all_train_y.extend(train_actinic_keratosis_y)

        all_actinic_keratosis = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/actinic_keratosis', [], width, height)
        test_actinic_keratosis_x, _ = np.split(all_actinic_keratosis, [int(len(all_actinic_keratosis))])

        test_actinic_keratosis_y = np.empty(len(test_actinic_keratosis_x))
        test_actinic_keratosis_y.fill(0)
        all_test_y.extend(test_actinic_keratosis_y)

        all_train_x.extend(train_actinic_keratosis_x)
        all_test_x.extend(test_actinic_keratosis_x)

        print(f'class {0} to {len(all_train_x)}')


        #Basal_cell_carcinoma
        all_basal_cell_carcinoma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/basal_cell_carcinoma', [], width, height)
        train_basal_cell_carcinoma_x, _ = np.split(all_basal_cell_carcinoma, [int(len(all_basal_cell_carcinoma)*percentage)])

        train_basal_cell_carcinoma_y = np.empty(len(train_basal_cell_carcinoma_x))
        train_basal_cell_carcinoma_y.fill(1)
        all_train_y.extend(train_basal_cell_carcinoma_y)

        all_basal_cell_carcinoma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/basal_cell_carcinoma', [], width, height)
        test_basal_cell_carcinoma_x, _ = np.split(all_basal_cell_carcinoma, [int(len(all_basal_cell_carcinoma))])

        test_basal_cell_carcinoma_y = np.empty(len(test_basal_cell_carcinoma_x))
        test_basal_cell_carcinoma_y.fill(1)
        all_test_y.extend(test_basal_cell_carcinoma_y)

        all_train_x.extend(train_basal_cell_carcinoma_x)
        all_test_x.extend(test_basal_cell_carcinoma_x)


        print(f'class {1} to {len(all_train_x)}')


        #Dermatofibroma
        all_dermatofibroma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/dermatofibroma', [], width, height)
        train_dermatofibroma_x, _ = np.split(all_dermatofibroma, [int(len(all_dermatofibroma)*percentage)])

        train_dermatofibroma_y = np.empty(len(train_dermatofibroma_x))
        train_dermatofibroma_y.fill(2)
        all_train_y.extend(train_dermatofibroma_y)

        all_dermatofibroma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/dermatofibroma', [], width, height)
        test_dermatofibroma_x, _ = np.split(all_dermatofibroma, [int(len(all_dermatofibroma))])

        test_dermatofibroma_y = np.empty(len(test_dermatofibroma_x))
        test_dermatofibroma_y.fill(2)
        all_test_y.extend(test_dermatofibroma_y)

        all_train_x.extend(train_dermatofibroma_x)
        all_test_x.extend(test_dermatofibroma_x)


        print(f'class {2} to {len(all_train_x)}')

        
        #Melanoma
        all_melanoma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/melanoma', [], width, height)
        train_melanoma_x, _ = np.split(all_melanoma, [int(len(all_melanoma)*percentage)])

        train_melanoma_y = np.empty(len(train_melanoma_x))
        train_melanoma_y.fill(3)
        all_train_y.extend(train_melanoma_y)

        all_melanoma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/melanoma', [], width, height)
        test_melanoma_x, _ = np.split(all_melanoma, [int(len(all_melanoma))])

        test_melanoma_y = np.empty(len(test_melanoma_x))
        test_melanoma_y.fill(3)
        all_test_y.extend(test_melanoma_y)

        all_train_x.extend(train_melanoma_x)
        all_test_x.extend(test_melanoma_x)



        print(f'class {3} to {len(all_train_x)}')


        #Nevus
        all_nevus = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/nevus', [], width, height)
        train_nevus_x, _ = np.split(all_nevus, [int(len(all_nevus)*percentage)])

        train_nevus_y = np.empty(len(train_nevus_x))
        train_nevus_y.fill(4)
        all_train_y.extend(train_nevus_y)

        all_nevus = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/nevus', [], width, height)
        test_nevus_x, _ = np.split(all_nevus, [int(len(all_nevus))])

        test_nevus_y = np.empty(len(test_nevus_x))
        test_nevus_y.fill(4)
        all_test_y.extend(test_nevus_y)

        all_train_x.extend(train_nevus_x)
        all_test_x.extend(test_nevus_x)



        print(f'class {4} to {len(all_train_x)}')


        #Pigmented_benign_keratosis
        all_pigmented_benign_keratosis = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/pigmented_benign_keratosis', [], width, height)
        train_pigmented_benign_keratosis_x, _ = np.split(all_pigmented_benign_keratosis, [int(len(all_pigmented_benign_keratosis)*percentage)])

        train_pigmented_benign_keratosis_y = np.empty(len(train_pigmented_benign_keratosis_x))
        train_pigmented_benign_keratosis_y.fill(5)
        all_train_y.extend(train_pigmented_benign_keratosis_y)

        all_pigmented_benign_keratosis = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/pigmented_benign_keratosis', [], width, height)
        test_pigmented_benign_keratosis_x, _ = np.split(all_pigmented_benign_keratosis, [int(len(all_pigmented_benign_keratosis))])

        test_pigmented_benign_keratosis_y = np.empty(len(test_pigmented_benign_keratosis_x))
        test_pigmented_benign_keratosis_y.fill(5)
        all_test_y.extend(test_pigmented_benign_keratosis_y)

        all_train_x.extend(train_pigmented_benign_keratosis_x)
        all_test_x.extend(test_pigmented_benign_keratosis_x)


        print(f'class {5} to {len(all_train_x)}')

        #Seborrheic_keratosis
        all_seborrheic_keratosis = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/seborrheic_keratosis', [], width, height)
        train_seborrheic_keratosis_x, _ = np.split(all_seborrheic_keratosis, [int(len(all_seborrheic_keratosis)*percentage)])

        train_seborrheic_keratosis_y = np.empty(len(train_seborrheic_keratosis_x))
        train_seborrheic_keratosis_y.fill(6)
        all_train_y.extend(train_seborrheic_keratosis_y)

        all_seborrheic_keratosis = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/seborrheic_keratosis', [], width, height)
        test_seborrheic_keratosis_x, _ = np.split(all_seborrheic_keratosis, [int(len(all_seborrheic_keratosis))])

        test_seborrheic_keratosis_y = np.empty(len(test_seborrheic_keratosis_x))
        test_seborrheic_keratosis_y.fill(6)
        all_test_y.extend(test_seborrheic_keratosis_y)

        all_train_x.extend(train_seborrheic_keratosis_x)
        all_test_x.extend(test_seborrheic_keratosis_x)


        print(f'class {6} to {len(all_train_x)}')


        #Squamous_cell_carcinoma
        all_squamous_cell_carcinoma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/squamous_cell_carcinoma', [], width, height)
        train_squamous_cell_carcinoma_x, _ = np.split(all_squamous_cell_carcinoma, [int(len(all_squamous_cell_carcinoma)*percentage)])

        train_squamous_cell_carcinoma_y = np.empty(len(train_squamous_cell_carcinoma_x))
        train_squamous_cell_carcinoma_y.fill(7)
        all_train_y.extend(train_squamous_cell_carcinoma_y)

        all_squamous_cell_carcinoma = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/squamous_cell_carcinoma', [], width, height)
        test_squamous_cell_carcinoma_x, _ = np.split(all_squamous_cell_carcinoma, [int(len(all_squamous_cell_carcinoma))])

        test_squamous_cell_carcinoma_y = np.empty(len(test_squamous_cell_carcinoma_x))
        test_squamous_cell_carcinoma_y.fill(7)
        all_test_y.extend(test_squamous_cell_carcinoma_y)

        all_train_x.extend(train_squamous_cell_carcinoma_x)
        all_test_x.extend(test_squamous_cell_carcinoma_x)


        print(f'class {7} to {len(all_train_x)}')


        #Vascular_lesion
        all_vascular_lesion = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Train/vascular_lesion', [], width, height)
        train_vascular_lesion_x, _ = np.split(all_vascular_lesion, [int(len(all_vascular_lesion)*percentage)])

        train_vascular_lesion_y = np.empty(len(train_vascular_lesion_x))
        train_vascular_lesion_y.fill(8)
        all_train_y.extend(train_vascular_lesion_y)

        all_vascular_lesion = self.load_images_from_folder('../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration/Test/vascular_lesion', [], width, height)
        test_vascular_lesion_x, _ = np.split(all_vascular_lesion, [int(len(all_vascular_lesion))])

        test_vascular_lesion_y = np.empty(len(test_vascular_lesion_x))
        test_vascular_lesion_y.fill(8)
        all_test_y.extend(test_vascular_lesion_y)

        all_train_x.extend(train_vascular_lesion_x)
        all_test_x.extend(test_vascular_lesion_x)


        print(f'class {8} to {len(all_train_x)}')


        test_x = np.array(all_test_x)
        test_y = np.array(all_test_y)
        train_x = np.array(all_train_x)
        train_y = np.array(all_train_y)

        prepped_train_x, prepped_train_y, prepped_test_x, prepped_test_y = self.prep_pixels(train_x, train_y, test_x, test_y)
        print("data loading complete...")
        print("shape: trx/try/tex/tey", prepped_train_x.shape, "/", prepped_train_y.shape, "/", prepped_test_x.shape, "/", prepped_test_y.shape)

        return prepped_train_x, prepped_train_y, prepped_test_x, prepped_test_y

    def convert_images_to_arr(self, images, all, pref_width = 32, pref_height = 32):
        res = []


        for i, img in enumerate(images):
            img = Image.open(img)
            rezised_img = img.resize((pref_width, pref_height))
            data = np.asarray(rezised_img)
            if data.shape == (pref_width, pref_height):
                d3 = np.stack((data,)*3, axis=-1)
                res.append(d3)
            else:
                res.append(data)

        # print(len(res))
        all.extend(res)

        return res
    
    def data_gen(self, images):
        training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    def load_images_from_folder(self, folder, all, pref_width = 32, pref_height = 32):
        res = []
        path = folder + "/*.jpg"
        print(f'\n\n{folder}\n\n')
        for img in glob.glob(path):
            res.append(img)

        return self.convert_images_to_arr(res, all, pref_width, pref_height)
    


    def prep_pixels(self, train_x, train_y, test_x, test_y):
        print("TrainY: ", train_y.shape, ", testY: ", test_y.shape)
        trainY = to_categorical(train_y)
        testY = to_categorical(test_y)

        # convert from integers to floats
        train_norm = train_x.astype('float32')
        test_norm = test_x.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, trainY, test_norm, testY

def __main__():
    DataUtils().api_load_medical_data(100, 100)


if __name__ == '__main__':
    __main__()