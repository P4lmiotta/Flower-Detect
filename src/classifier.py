import os
import cv2 as cv
import numpy as np
import h5py

# la libreria dask può essere utilizzata nel caso in cui
# vengono manipolati array di dimensione non sostenibile da un array della libreria numpy
import dask.array as da

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import src.extract_features
import src.data_augmentation

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
                            classification_report


class Classifier:
    def __init__(self):
        self.X = []

        self.Y = []
        self.le = LabelEncoder()

        self.DATASETS_DIR = "C:\\Users\\pc\\PycharmProjects\\flower_detect\\datasets\\flowers"
        self.TEST_DIR = "C:\\Users\\pc\\PycharmProjects\\flower_detect\\test_images\\test_"
        self.train_labels = os.listdir(self.DATASETS_DIR)
        self.test_labels = os.listdir(self.TEST_DIR)

    # estraggo le features
    def make_train_data(self):
        resize = tuple((500, 500))
        for training_name in self.train_labels:
            label = training_name
            DIR = os.path.join(self.DATASETS_DIR, training_name)

            for img in tqdm(os.listdir(DIR)):
                path = os.path.join(DIR, img)
                img = cv.imread(path)
                img = cv.resize(img, resize)

                glob_features_b = src.extract_features.extract(img)

                self.X.append(glob_features_b)
                self.Y.append(str(label))

    def pass_train_data(self):
        self.make_train_data()
        print(len(self.X))

    # normalizzo le features estratte
    def normalize_features(self):
        scaler = MinMaxScaler(feature_range=(0, 1))

        dask_array_compute = da.asarray(self.X) #np.array(self.X)
        rescaled_features = scaler.fit_transform(dask_array_compute)

        target = self.le.fit_transform(self.Y)

        return rescaled_features, target

    # salvo le features utilizzando la libreria 'h5py'
    def feature_vector(self):
        print("Normalize features...")
        rescaled_features, target = self.normalize_features()

        print("Passing features normalized into dataset files h5py...")
        h5f_data = h5py.File('C:\\Users\\pc\PycharmProjects\\pythonProject\\flowers_detection\\datasets_file\\data.h5',
                             'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

        h5f_label = h5py.File(
            'C:\\Users\\pc\PycharmProjects\\pythonProject\\flowers_detection\\datasets_file\\labels.h5', 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))

        h5f_data.close()
        h5f_label.close()

        return h5f_data, h5f_label

    # converto le features salvate nei file .h5 in stringhe
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

    # definisco i migliori hyperparameters da passare nel modello ML utilizzato
    def define_hyperparameters(self):
        hyperparameters_grid_tree = {
            'bootstrap': [True],
            'max_depth': [20],
            'max_features': ['auto'],
            'min_samples_leaf': [2, 3],
            'min_samples_split': [4],
            'n_estimators': [200, 500, 800]
        }

        return hyperparameters_grid_tree

    def training_test(self):
        global_features, global_labels = self.feature_to_string()

        # split il training e il test set
        train_X, test_X, train_Y, test_Y = train_test_split(np.array(global_features), np.array(global_labels),
                                                            test_size=0.25,
                                                            random_state=42)
        # prendo i parametri per il GridSearchCV
        param_grid = self.define_hyperparameters()

        # creo il modello - Gaussian NB
        gnb = GaussianNB()

        # addestro il modello tramite il training set
        gnb.fit(train_X, train_Y)

        # creo il modello - Random Forest
        forest_clf = RandomForestClassifier()

        # creo il modello - GridSearchCV - Random Forest
        forest_grid_search = GridSearchCV(estimator=forest_clf, param_grid=param_grid,
                                          cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)

        # addestro il modello tramite il training set
        forest_grid_search.fit(train_X, train_Y)

        # creo il modello - Extra Trees
        extra_clf = ExtraTreesClassifier()

        # creo il modello - GridSearchCV - Extra Trees
        extra_grid_search = GridSearchCV(estimator=extra_clf, param_grid=param_grid,
                                         cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)

        # addestro il modello tramite il training set
        extra_grid_search.fit(train_X, train_Y)

        print("---------------------------")
        print("---------------------------")

        self.X = np.array(self.X)

        # visualizzo gli scores del test set

        grid_predictionRF_test = forest_grid_search.predict(test_X)
        print("Scores Random Forest: ")
        accuracy = accuracy_score(test_Y, grid_predictionRF_test)
        precision = precision_score(test_Y, grid_predictionRF_test, average='macro')
        recall = recall_score(test_Y, grid_predictionRF_test, average='macro')
        f1 = f1_score(test_Y, grid_predictionRF_test, average='macro')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("f1 measure:", f1)
        print("\n\n")

        print("Best parameters for Random Forest model: ")
        print(forest_grid_search.best_params_)

        print("Classification report Random Forest model: ")
        print(classification_report(test_Y, grid_predictionRF_test))

        print("---------------")

        grid_prediction_test = extra_grid_search.predict(test_X)
        print("Scores Extra Trees: ")
        accuracy = accuracy_score(test_Y, grid_prediction_test)
        precision = precision_score(test_Y, grid_prediction_test, average='macro')
        recall = recall_score(test_Y, grid_prediction_test, average='macro')
        f1 = f1_score(test_Y, grid_prediction_test, average='macro')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("f1 measure:", f1)
        print("\n\n")

        print("Best parameters for Extra Tree model: ")
        print(extra_grid_search.best_params_)

        print("---------------")

        gnb_prediction_test = gnb.predict(test_X)
        print("Scores GaussianNB: ")
        accuracy = accuracy_score(test_Y, gnb_prediction_test)
        precision = precision_score(test_Y, gnb_prediction_test, average='macro')
        recall = recall_score(test_Y, gnb_prediction_test, average='macro')
        f1 = f1_score(test_Y, gnb_prediction_test, average='macro')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("f1 measure:", f1)
        print("\n\n")

        print("---------------")
        print("---------------")

        print("Display confusion matrix for best model:")
        self.display_cm(test_Y, grid_predictionRF_test)

        return forest_grid_search  # best model

    def display_cm(self, test_Y, clf_prediction):
        # Plot non-normalized confusion matrix
        target_labels = np.unique(self.Y)

        matrix = confusion_matrix(test_Y, clf_prediction)
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=[target for target in target_labels],
                    yticklabels=[target for target in target_labels],
                    annot=True,
                    fmt='d')
        plt.title('Confusion Matrix RFC')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


    # metodo che permette di visualizzare predizioni di una serie di immagini
    def testing_images(self, forest_grid_search):
        resize = tuple((500, 500))
        for test_name in self.test_labels:
            label_test = test_name
            DIR_TEST = os.path.join(self.TEST_DIR, test_name)

            for img in tqdm(os.listdir(DIR_TEST)):
                path = os.path.join(DIR_TEST, img)
                img = cv.imread(path)
                if img is not None:
                    img = cv.resize(img, resize)

                glob_features_p = src.extract_features.extract(img)

                prediction = forest_grid_search.predict(glob_features_p.reshape(1, -1))[0]

                cv.putText(img, 'Predict flower: {}'.format(self.train_labels[prediction]), (20, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 116, 55), 3)

                cv.putText(img, 'Actual flower: {}'.format(str(label_test)), (20, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 116, 55), 3)

                # visualizza l'immagine
                plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                plt.show()


# Driver Code
if __name__ == '__main__':
    classifier = Classifier()

    print("Extracting features from datasets...")
    classifier.pass_train_data()
    print("Extract completed!")

    print('--------------------')

    clf = classifier.training_test()
    print("fit completed!")

    print('--------------------')

    print("Test some images... with the best model used!")
    classifier.testing_images(clf)
    print("Test completed! Let's visualize the image predictions!")
