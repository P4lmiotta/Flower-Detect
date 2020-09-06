import os
import cv2 as cv
import numpy as np
import h5py
import dask.array as da
from tqdm import tqdm
import matplotlib.pyplot as plt

import src.extract_features

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report


class Classifier:
    def __init__(self):
        self.X = []

        self.Y = []
        self.le = LabelEncoder()

        self.DATASETS_DIR = "C:\\Users\\pc\\PycharmProjects\\pythonProject\\flowers_detection\\datasets\\flowers"
        self.TEST_DIR = "C:\\Users\\pc\\PycharmProjects\\pythonProject\\flowers_detection\\test_images\\test"
        self.train_labels = os.listdir(self.DATASETS_DIR)
        self.test_labels = os.listdir(self.TEST_DIR)

    def make_train_data(self):
        for training_name in self.train_labels:
            label = training_name
            DIR = os.path.join(self.DATASETS_DIR, training_name)

            for img in tqdm(os.listdir(DIR)):
                path = os.path.join(DIR, img)
                img = cv.imread(path)

                glob_features_b = src.extract_features.extract(img)

                self.X.append(glob_features_b)
                self.Y.append(str(label))

    def pass_train_data(self):
        self.make_train_data()
        print(len(self.X))

    def normalize_features(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dask_array_compute = da.asarray(self.X)  # non eseguire se non si usa f_grad
        rescaled_features = scaler.fit_transform(dask_array_compute)  # self.X senza f_grad

        target_names = np.unique(self.Y)
        print(target_names)

        target = self.le.fit_transform(self.Y)
        target_u = np.unique(self.Y)
        print(target_u)

        return rescaled_features, target

    # save the feature vector using HDF5
    def feature_vector(self):
        rescaled_features, target = self.normalize_features()

        h5f_data = h5py.File('C:\\Users\\pc\PycharmProjects\\pythonProject\\flowers_detection\\datasets_file\\data.h5',
                             'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

        h5f_label = h5py.File(
            'C:\\Users\\pc\PycharmProjects\\pythonProject\\flowers_detection\\datasets_file\\labels.h5', 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))

        h5f_data.close()
        h5f_label.close()

        return h5f_data, h5f_label

    def feature_to_string(self):
        h5f_data, h5f_label = self.feature_vector()
        h5f_data = h5py.File('C:\\Users\\pc\PycharmProjects\\pythonProject\\flowers_detection\\datasets_file\\data.h5',
                             'r')
        h5f_label = h5py.File(
            'C:\\Users\\pc\PycharmProjects\\pythonProject\\flowers_detection\\datasets_file\\labels.h5', 'r')

        global_features_string = h5f_data['dataset_1']
        global_labels_string = h5f_label['dataset_1']

        global_features = np.array(global_features_string)
        global_labels = np.array(global_labels_string)
        return global_features, global_labels

    def define_hyperparameters(self):
        hyperparameters_grid = {
            'bootstrap': [True],
            'max_depth': [20],
            'max_features': [3],
            'min_samples_leaf': [2],
            'min_samples_split': [4],
            'n_estimators': [300]
        }

        return hyperparameters_grid

    def training_test(self):
        global_features, global_labels = self.feature_to_string()

        # split il training e il test set
        train_X, test_X, train_Y, test_Y = train_test_split(np.array(global_features), np.array(global_labels),
                                                            test_size=0.20,
                                                            random_state=42)
        # prendo i parametri per il GridSearchCV
        param_grid = self.define_hyperparameters()

        # creo il modello - Random Forest
        forest_clf = RandomForestClassifier()

        # creo il modello - GridSearchCV
        forest_grid_search = GridSearchCV(estimator=forest_clf, param_grid=param_grid, cv=StratifiedKFold(n_splits=3),
                                          n_jobs=-1, verbose=2)

        # creo il modello - RandomizedSearchCV
        #forest_random_search = RandomizedSearchCV(estimator=forest_clf, param_distributions=param_grid,
                                                  #cv=StratifiedKFold(n_splits=3),
                                                  #n_jobs=-1, verbose=2, scoring='accuracy')

        # addestro il modello tramite il training set
        forest_grid_search.fit(train_X, train_Y)
        print("---------------------------")
        #forest_random_search.fit(train_X, train_Y)

        print("---------------------------")


        self.X = np.array(self.X)
        grid_prediction = forest_grid_search.predict(train_X)
        print("evaluation training data GridSearchCV:")
        print(classification_report(train_Y, grid_prediction))

        grid_prediction_test = forest_grid_search.predict(test_X)
        print("evaluation test data GridSearchCV:")
        print(classification_report(test_Y, grid_prediction_test))
        '''
        random_prediction = forest_grid_search.predict(train_X)
        print("evaluation training data RandomizedSearchCV:")
        print(classification_report(train_Y, random_prediction))

        random_prediction_test = forest_grid_search.predict(test_X)
        print("evaluation test data RandomizedSearchCV:")
        print(classification_report(test_Y, random_prediction_test))
        '''
        return forest_grid_search

    def testing_images(self, forest_grid_search):
        resize = tuple((500, 500))

        for test_name in self.test_labels:
            label_test = test_name
            DIR_TEST = os.path.join(self.TEST_DIR, test_name)

            for img in os.listdir(DIR_TEST):
                path = os.path.join(DIR_TEST, img)
                img = cv.imread(path)
                if img is not None:
                    img = cv.resize(img, resize)

                f_moments = src.extract_features.fd_hu_moments(img)
                f_haralick = src.extract_features.fd_haralick(img)
                f_istogramm = src.extract_features.fd_histogram(img)

                global_features = np.hstack([f_moments, f_haralick, f_istogramm])

                prediction = forest_grid_search.predict(global_features.reshape(1, -1))[0]

                cv.putText(img, 'Predict flower: {}'.format(self.train_labels[prediction]), (20, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 116, 55), 3)

                cv.putText(img, 'Actual flower: {}'.format(str(label_test)), (20, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 116, 55), 3)

                # visualizza l'immagine
                plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                plt.show()


if __name__ == '__main__':
    classifier = Classifier()

    classifier.pass_train_data()

    print('--------------------')
    classifier.training_test()

    #forest_grid_search = classifier.training_test()
    #classifier.testing_images(forest_grid_search)
