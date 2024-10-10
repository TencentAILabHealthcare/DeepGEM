import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix


def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating.
    https://github.com/benhamner/Metrics
    https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
    """
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum()
    O = O / O.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return 1 - (num / den)


"""
# cohen_kappa_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
>>> from sklearn.metrics import cohen_kappa_score
>>> y_true = [2, 0, 2, 2, 0, 1]
>>> y_pred = [0, 0, 2, 2, 0, 2]
"""


def c_index(labels, hazards):
    labels = labels.reshape(-1, 2)

    hazards = hazards.reshape(-1)
    hazard = []
    label = []  # event
    surv_time = []  # duration
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            surv_time.append(labels[i, 0])
            label.append(labels[i, 1])
            hazard.append(hazards[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return concordance_index(new_surv, -new_hazard, new_label)
