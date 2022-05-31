from tensorflow.keras.utils import to_categorical
import glob 
import numpy as np
from PIL import Image
import os


class DataUtils():
    def api_load_medical_data(self, width=32, height = 32):
        all_test_x = []
        all_test_y = []
        all_train_x = []
        all_train_y = []

        #COVID
        test_cov_x = self.load_images_from_folder('../data/COVID-19_Radiography_Dataset/COVID_test', all_test_x, width, height)
        test_cov_y = np.empty(len(test_cov_x))
        test_cov_y.fill(0)
        all_test_y.extend(test_cov_y)

        train_cov_x = self.load_images_from_folder('../data/COVID-19_Radiography_Dataset/COVID_train', all_train_x, width, height)
        train_cov_y = np.empty(len(train_cov_x))
        train_cov_y.fill(0)
        all_train_y.extend(train_cov_y)

        #NORMAL
        test_normal_x = self.load_images_from_folder('../data/COVID-19_Radiography_Dataset/NORMAL_test', all_test_x, width, height)
        test_normal_y = np.empty(len(test_normal_x))
        test_normal_y.fill(1)
        all_test_y.extend(test_normal_y)

        train_normal_x = self.load_images_from_folder('../data/COVID-19_Radiography_Dataset/NORMAL_train', all_train_x, width, height)
        train_normal_y = np.empty(len(train_normal_x))
        train_normal_y.fill(1)
        all_train_y.extend(train_normal_y)

        #Pneu
        test_pneu_x = self.load_images_from_folder('../data/COVID-19_Radiography_Dataset/PNEMONIA_test', all_test_x, width, height)
        test_pneu_y = np.empty(len(test_pneu_x))
        test_pneu_y.fill(2)
        all_test_y.extend(test_pneu_y)

        train_pneu_x = self.load_images_from_folder('../data/COVID-19_Radiography_Dataset/PNEMONIA_train', all_train_x, width, height)
        train_pneu_y = np.empty(len(train_pneu_x))
        train_pneu_y.fill(2)
        all_train_y.extend(train_pneu_y)

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

        all.extend(res)

        return res

    def load_images_from_folder(self, folder, all, pref_width = 32, pref_height = 32):
        res = []
        path = folder + "/*.png"
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
    DataUtils().api_load_medical_data()


if __name__ == '__main__':
    __main__()