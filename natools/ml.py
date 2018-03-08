from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC

from natools.general import ProgressBar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import natools.general as general_utils

import pickle


def get_data_pre_processing_object(data, standardization=True, normalization=True):
    """
    :param data: numpy array of shape (num_samples, num_features)
    :param standardization: boolean
    :param normalization: boolean
    :return: scikit learn Pipeline() object with the transform() method
    
    Useful to have the object fitted on training data first, to then apply it
    to test data.
    """
    steps = list()
    if standardization:
        steps.append(('standardize', StandardScaler()))
    if normalization:
        steps.append(('normalize', Normalizer()))
    if len(steps) == 0:
        return None
    else:
        clf = Pipeline(steps)
        clf.fit(data)
        return clf


def pre_process_data(data, standardization=True, normalization=True):
    """
    :param data: numpy array of shape (num_samples, num_features)
    :param standardization: boolean
    :param normalization: boolean
    :return: transformed data
    """
    clf = get_data_pre_processing_object(data, standardization=standardization, normalization=normalization)
    if clf is None:
        return data
    else:
        return clf.transform(data)


def split_train_test(x, y, indices=None, stratification_values=None,
                     test_size=0.25, random=True):
    """
    In order to keep track of the indices, can either input pandas
    DataFrame() objects or give the indices object to the function.

    The stratification_values ensure that the train and test datasets keep
    the proportions of the categories in the input data.

    Random state ensures (if True) that the splitting is done randomly (
    useful for production) but can also ensure (if False) that the splitting
    remains the same within a session. The latter is useful when testing out
    stuff.

    TO DO:
    1) allow for stratification and grouping...
    """
    if random:
        random = None
    else:
        random = 42

    if indices is None:
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=test_size, random_state=random,
                             stratify=stratification_values)
    else:
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, indices, test_size=test_size,
                             random_state=random,
                             stratify=stratification_values)

    return x_train, x_test, y_train, y_test


def cross_validate_model(model, x_train, y_train, folds=10,
                         standardization=False, normalization=False,
                         stratification=False, grouping=None, shuffle=False,
                         metric=None, verbose=True):
    """
    x_train and y_train can be pandas DataFrame() objects or numpy arrays

    model: could be for example RandomForestClassifier(n_estimators=50)

    stratification: specify on what to stratify (usually label to keep
    proportions, otherwise on some other characteristic of the data)

    grouping: to make sure data from same group don't end up in both the
    training and the testing folds (e.g. if have users, and several labels
    produced from each user). If classes are well balanced, the stratification
    should be almost naturally respected

    shuffling: important if the data is very time-dependent

    metric: if nothing specified, the model's default metric will be used

    TO DO:
    1) make stratification and grouping work together (will be approximate)
    2) have stratification work with any label (not only the classes)
    3) make shuffling and grouping work together
    """

    if stratification and (grouping is not None):
        raise ValueError("not supporting stratification and grouping yet...")
    if (grouping is not None) and shuffle:
        raise ValueError("not supporting grouping and shuffling yet...")
    if folds < 2:
        raise ValueError("not enough folds...")

    steps = list()
    if standardization:
        steps.append(('standardize', StandardScaler()))
    if normalization:
        steps.append(('normalize', Normalizer()))
    steps.append(('classify', model))
    clf = Pipeline(steps)

    if 'pandas' in str(type(x_train)):
        x_train = x_train.values
    if 'pandas' in str(type(y_train)):
        y_train = y_train.values.ravel()
    else:
        y_train = y_train.ravel()

    cv = None
    if stratification:
        cv = StratifiedKFold(n_splits=folds, shuffle=shuffle)
    if grouping is not None:
        cv = GroupKFold(folds).split(x_train, y_train, grouping)
    if cv is None:
        cv = folds

    scores = cross_val_score(clf, x_train, y_train, cv=cv, scoring=metric)

    if verbose:
        print("Metric: %0.2f (+/- %0.2f)"
              % (scores.mean(), scores.std() * 2))

    return scores


def get_trained_model(model, x_train, y_train, standardization=True,
                      normalization=True):
    steps = list()
    if standardization:
        steps.append(('standardize', StandardScaler()))
    if normalization:
        steps.append(('normalize', Normalizer()))
    steps.append(('classify', model))
    clf = Pipeline(steps)

    if 'pandas' in str(type(x_train)):
        x = x_train.values
    else:
        x = x_train
    if 'pandas' in str(type(y_train)):
        y = y_train.values.ravel()
    else:
        y = y_train.ravel()

    clf.fit(x, y)
    return clf


def get_pairwise_distances(data):
    distances = np.array([]).reshape(0, data.shape[0])
    length = data.shape[0]
    p = general_utils.ProgressBar(length)
    count = 0
    for sample_1 in data:
        row = []
        for sample_2 in data:
            row.append(general_utils.compute_euclidean_distance(sample_1, sample_2))
        distances = np.concatenate((distances, np.array([row])))
        if length > 1:
            # Update progress bar
            count += 1
            p.animate(count)
    return distances


def get_training_set_2d_coordinates(distance_matrix, labels, random_state=None):
    """
    Other approach: t-SNE?
    """
    training_coordinates = MDS(n_components=2, random_state=random_state,
                               dissimilarity='precomputed')
    training_coordinates.fit(distance_matrix)

    df = pd.DataFrame(dict(x=training_coordinates.embedding_[:, 0],
                           y=training_coordinates.embedding_[:, 1],
                           label=labels))
    # note: the stress_ is the sum of squared distance of the
    # disparities and the distances for all constrained points
    return df, training_coordinates.stress_


def visualize_training_set(df, disparities, categories=None, colors=None,
                           plot_margin=0.05, marker_size=5, alpha=0.7,
                           marker='o', legend_loc='upper right',
                           legend_size=7, title_size=11, title=None,
                           image_path=None):
    """
    More accurate if the distance metric is an actual distance metric
    (triangle inequality, etc.). Otherwise the between-object distances
    are preserved as well as possible.
    [takes an input matrix giving dissimilarities between pairs of items
    and outputs a coordinate matrix whose configuration minimizes a loss
    function]

    Method: multidimensional scaling (MDS)
    """
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(plot_margin)
    for lbl, group in groups:
        if categories is not None:
            name = categories[lbl]
        else:
            name = lbl
        if colors is not None:
            color = colors[lbl]
        else:
            color = None
        ax.plot(group.x, group.y, marker=marker, linestyle='',
                ms=marker_size,
                label=name, c=color, alpha=alpha)
    ax.legend(loc=legend_loc, prop={'size': legend_size})
    if disparities is not None:
        disparities = disparities / df.shape[0]
        title_addition = "Disparities: %d\n" % disparities
    else:
        title_addition = "Disparity level not available..."
    if title is None:
        title = title_addition
    else:
        title = title + "\t[" + title_addition + "]"
    plt.title(title, fontsize=title_size)
    if image_path is not None:
        plt.savefig(image_path)
    plt.show()


class GmmClassification:
    def __init__(self, n_components=2, variance='spherical'):
        self.variance = variance
        self.n_components = n_components
        self.x_train = None
        self.gmm = None
        self.labels_ = None

    def fit(self, x_train):
        self.x_train = x_train
        gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.variance)
        gmm.fit(x_train)
        self.gmm = gmm
        self.labels_ = self.predict(x_train)
        return self

    def predict(self, x_test):
        if self.gmm is None:
            raise AttributeError("the model has to be trained before testing!")
        else:
            return np.argmax(self.gmm.predict_proba(x_test), axis=1)

    def score(self, x_test, y_test):
        """
        Classical accuracy score.
        """
        predictions = self.predict(x_test)
        return accuracy_score(y_test, predictions)

    def save(self, path):
        """
        careful: cannot handle lambda functions, if such functions were
        passed to the initial constructor...
        """
        file = open(path, 'wb')
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        return pickle.load(file)


class ClassificationFromClustering:
    def __init__(self, epsilon, min_samples, kernel, C, degree=None):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.x_train = None
        self.y_train = None
        self.clusterer = None
        self.classifier = None

    def fit(self, x_train):
        self.x_train = x_train
        model = DBSCAN(eps=self.epsilon, min_samples=self.min_samples,
                       algorithm='ball_tree')
        model.fit(x_train)
        self.clusterer = model
        self.y_train = model.labels_

        clf = SVC(kernel=self.kernel, C=self.C, degree=self.degree)
        clf.fit(self.x_train, self.y_train)
        self.classifier = clf

        return self

    def predict(self, x_test):
        if self.classifier is None:
            raise AttributeError("the model has to be trained before testing!")
        else:
            return self.classifier.predict(x_test)

    def plot_boundary(self, x_test):
        h = .02  # step size in the mesh
        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=self.predict(x_test),
                    cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()

    def score(self, x_test, y_test):
        """
        Classical accuracy score.
        """
        predictions = self.predict(x_test)
        return accuracy_score(y_test, predictions)

    def save(self, path):
        """
        careful: cannot handle lambda functions, if such functions were
        passed to the initial constructor...
        """
        file = open(path, 'wb')
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        return pickle.load(file)


class SupervisedGMM:
    """Initialize with decision_boundary=None to have the full optimisation
    (careful, it takes a long time)"""
    def __init__(self, variance='spherical', n_components_0=1,
                 n_components_1=1, decision_boundary=0.5, zero_value=0,
                 one_value=1):
        self.variance = variance
        self.n_components_0 = n_components_0
        self.n_components_1 = n_components_1
        self.decision_boundary = decision_boundary  # relative to the positive class
        self.zero_value = zero_value
        self.one_value = one_value
        self.gmm_0 = None
        self.gmm_1 = None
        self.x_train = None
        self.y_train = None
        self.min_0 = None
        self.max_0 = None
        self.min_1 = None
        self.max_1 = None

    def fit(self, x_train, y_train, verbose=False):
        # This data will also be stored in each gmm component, but we keep
        # it for convenience
        self.x_train = x_train
        self.y_train = y_train

        gmm_0 = GaussianMixture(n_components=self.n_components_0,
                                covariance_type=self.variance)
        gmm_0.fit(x_train[y_train == self.zero_value])
        self.gmm_0 = gmm_0

        gmm_1 = GaussianMixture(n_components=self.n_components_1,
                                covariance_type=self.variance)
        gmm_1.fit(x_train[y_train == self.one_value])
        self.gmm_1 = gmm_1

        # Builds the probability structure, optimizes the decision boundary
        scores = pd.DataFrame(columns=["score_0", "score_1"])
        j = 0
        for seq in self.x_train:
            score_0 = self.gmm_0.score_samples(seq.reshape(1, -1))
            score_1 = self.gmm_1.score_samples(seq.reshape(1, -1))
            scores.loc[j, :] = [score_0, score_1]
            j += 1
        """Try looking at the 95 or 99% percentiles, and the 0.01 and 0.05% percentiles instead of the max and min, to be resilient to outliers?"""
        self.min_0, self.max_0 = \
            scores["score_0"].min(), scores["score_0"].max()
        self.min_1, self.max_1 = \
            scores["score_1"].min(), scores["score_1"].max()

        if self.decision_boundary is None:
            probabilities = self.predict_proba(self.x_train)
            df = pd.DataFrame(columns=["boundary", "metric"])
            j = 0
            n = probabilities.shape[0] - 1
            p = ProgressBar(n)
            for i in range(n):
                boundary = 0.5 * (probabilities[i, 1] + probabilities[i+1, 1])
                self.decision_boundary = boundary
                df.loc[j, :] = [boundary, self.score(self.x_train,
                                                     self.y_train)]
                j += 1
                p.animate(i)
            print()
            self.decision_boundary = \
                df.loc[df['metric'].argmax(), "boundary"]

        if verbose:
            print("Decision boundary:", str(self.decision_boundary))

        return self

    def predict(self, x_test):
        if self.gmm_0 is None:
            raise AttributeError("the model has to be trained before testing!")
        else:
            probabilities = self.predict_proba(x_test)
            predictions = []
            for prob in probabilities[:, 1]:
                """
                We don't like False Positives (work detection), so if it's
                exactly on the boundary we prefer to put it in the negative
                class...
                """
                if prob > self.decision_boundary:
                    predictions.append(1)
                else:
                    predictions.append(0)
            predictions = np.array(predictions)
            return predictions

    def predict_proba(self, x_test):
        if self.gmm_0 is None:
            raise AttributeError("the model has to be trained before testing!")
        else:
            probabilities = np.array([]).reshape(-1, 2)
            for seq in x_test:
                score_0 = self.gmm_0.score_samples(seq.reshape(1, -1))
                prob_0 = (score_0 - self.min_0) / (self.max_0 - self.min_0)
                prob_0 = prob_0 if 0 <= prob_0 <= 1 else 0 if prob_0 < 0 else 1

                score_1 = self.gmm_1.score_samples(seq.reshape(1, -1))
                prob_1 = (score_1 - self.min_1) / (self.max_1 - self.min_1)
                prob_1 = prob_1 if 0 <= prob_1 <= 1 else 0 if prob_1 < 0 else 1

                prob = 0.5 * (prob_1 + 1 - prob_0)
                probabilities = \
                    np.concatenate((probabilities,
                                    np.array([1-prob, prob]).reshape(1, 2)))
            return probabilities

    def score(self, x_test, y_test):
        """
        Classical accuracy score.
        """
        predictions = self.predict(x_test)
        return accuracy_score(y_test, predictions)

    def save(self, path):
        """
        careful: cannot handle lambda functions, if such functions were
        passed to the initial constructor...
        """
        file = open(path, 'wb')
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        return pickle.load(file)


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False,
             hide_threshold=None):
    """
    Pretty print for confusion matrices...

    Rows represent the ground truth, columns represent the prediction...

    Example: cm = confusion_matrix([1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1, 1, 0], [0, 1])
    print_cm(cm, ["non work", "work"], hide_zeroes=True)
    """
    columnwidth = max([len(x) for x in labels] + [10])  # 10 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end='')
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end='')
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end='')
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end='')
        print()


def reliability_curve(y_true, y_score, bins=10, normalize=False):
    """Compute reliability curve

    Reliability curves allow checking if the predicted probabilities of a
    binary classifier are well calibrated. This function returns two arrays
    which encode a mapping from predicted probability to empirical probability.
    For this, the predicted probabilities are partitioned into equally sized
    bins and the mean predicted probability and the mean empirical probabilties
    in the bins are computed. For perfectly calibrated predictions, both
    quantities whould be approximately equal (for sufficiently many test
    samples).

    Note: this implementation is restricted to binary classification.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels (0 or 1).

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values. If normalize is False, y_score must be in
        the interval [0, 1]

    bins : int, optional, default=10
        The number of bins into which the y_scores are partitioned.
        Note: n_samples should be considerably larger than bins such that
              there is sufficient data in each bin to get a reliable estimate
              of the reliability

    normalize : bool, optional, default=False
        Whether y_score needs to be normalized into the bin [0, 1]. If True,
        the smallest value in y_score is mapped onto 0 and the largest one
        onto 1.


    Returns
    -------
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.


    References
    ----------
    .. [1] `Predicting Good Probabilities with Supervised Learning
            <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

    """
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    counts = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
        counts[i] = np.sum(y_true[bin_idx])
    return y_score_bin_mean, empirical_prob_pos, counts


if __name__ == "__main__":
    pass
