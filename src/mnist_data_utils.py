import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets


def get_data():
    df = pd.read_csv('../data/HAM/archive/hmnist_28_28_RGB.csv')

    n_samples = len(df.index)
    images = np.array(df.drop(['label'],axis=1))
    images = images/255.0
    images = images.reshape(n_samples,28,28,3)

    labels = np.array(df['label'].values)

    print(images.shape)

    y = df['label'].values
    X = df.drop(['label'] , axis=1).values

    X = X/255
    num_classes = 25

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    print(f'X_train/X_test/y_train/y_test: {X_train.shape}/{X_test.shape}/{y_train.shape}/{y_test.shape}')

    return X_train, X_test, y_train, y_test

def main():
    get_data()

if __name__ == '__main__':
    main()