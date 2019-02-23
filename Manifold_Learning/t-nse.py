# coding: utf-8
import multiprocessing

import numpy as np
from matplotlib import pyplot as plt 
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold # K-分割交差検証
from sklearn.ensemble import RandomForestClassifier 
from MulticoreTSNE import MulticoreTSNE as TSNE  # マルチコアで高速なt-SNE(このデータで約１分)


def main():
    # 手書き文字(画像粗め)
    dataset = datasets.load_digits() #(64, 1797)
    X = dataset.data 
    y = dataset.target 

    manifolder = TSNE(n_jobs=multiprocessing.cpu_count(), n_components=2)

    plot_embedding(X, y, manifolder)

def plot_embedding(X, y, manifolder):
    X = manifolder.fit_transform(X)
    for data, label in zip(X, y):
        plt.text(*X, str(y))
    plt.show()

def classify(X, y, manifolder):
    cross_validator = KFold(n_splits=10)
    scores = np.array([], dtype=np.bool)
    for train, test in cross_validator.split(X):
        X_trans = manifolder.fit_transform(X) # 次元縮約
        X_train, X_test = X_trans[train], X_trans[test]
        y_train, y_test = y[train], y[test]
        cls = RandomForestClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        scores = np.hstack((scores, y_test==y_pred))
        accuracy_score = sum(scores) / len(scores)
    print(accuracy_score)

if __name__ == '__main__':
    main()
    # 0.979410127991