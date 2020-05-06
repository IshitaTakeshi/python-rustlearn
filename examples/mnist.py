import numpy as np

from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from pyrustlearn import SGDClassifier


def _get_data():

    data = load_digits()

    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    return (X, y)


def run_example():

    data, target = _get_data()

    n_splits = 5
    accuracy = 0.0

    kf = KFold(n_splits=n_splits, shuffle=True)

    for (train_idx, test_idx) in kf.split(data):

        train_X = data[train_idx]
        train_y = target[train_idx]

        test_X = data[test_idx]
        test_y = target[test_idx]

        model = SGDClassifier()
        model.fit(train_X, train_y)

        predictions = model.predict(test_X)

        accuracy += accuracy_score(predictions, test_y)

    return accuracy / n_splits


accuracy = run_example()


print('Accuracy %s' % accuracy)
