import numpy as np
from sklearn import metrics
from statsmodels.distributions.empirical_distribution import ECDF


def ABCC(y_pred, y_gt, sensitive_attribute):
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    sensitive_attribute = sensitive_attribute.ravel()

    y_pre_1 = y_pred[sensitive_attribute == 1]
    y_pre_0 = y_pred[sensitive_attribute == 0]

    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    x = np.linspace(0, 1, 10000)
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # area under the lower kde, from the first leftmost point to the first intersection point
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)
    abcc *= 100

    return abcc



def MCDP_exact(y_pred, y_gt, s, epsilon=0.01):
    y_pred, y_gt, s = y_pred.ravel(), y_gt.ravel(), s.ravel()
    y_pre_1, y_pre_0 = y_pred[s==1], y_pred[s==0]
    ecdf0, ecdf1 = ECDF(y_pre_0), ECDF(y_pre_1)
    if epsilon == 0:
        mcdp = np.max(np.abs(ecdf0(y_pred) - ecdf1(y_pred)))*100
    else:
        y_pred = np.r_[0,np.sort(y_pred),1]
        mcdp = np.min(y_pred[y_pred <= epsilon])
        y_arr = y_pred[y_pred <= 1-epsilon].reshape((-1,1)) # yi,i\in\mathcal{I}
        delta_ecdf = np.r_[1.1, np.abs(ecdf0(y_pred) - ecdf1(y_pred))]

        selected_indices = np.where(y_arr <= y_pred,1,0) * np.where(y_arr+2*epsilon >= y_pred, 1, 0)
        selected_indices = selected_indices.reshape(-1) * np.tile(np.arange(len(y_pred))+1,len(y_arr))
        selected_indices = selected_indices.reshape((len(y_arr),len(y_pred)))
        delta_ecdf = delta_ecdf[selected_indices]
        mcdp = max(mcdp, np.max(np.min(delta_ecdf,axis=1)))*100
    return mcdp



def MCDP_approximate(y_pred, y_gt, s, epsilon_list, K=1):
    y_pred, y_gt, s = y_pred.ravel(), y_gt.ravel(), s.ravel()
    y_pre_1, y_pre_0 = y_pred[s==1], y_pred[s==0]
    ecdf0, ecdf1 = ECDF(y_pre_0), ECDF(y_pre_1)
    mcdp_list = []

    for epsilon in epsilon_list:
        delta = epsilon / K
        mcdp = np.min(np.abs(ecdf0(np.arange(K+1)*delta) - ecdf1(np.arange(K+1)*delta)))
        probing_points = np.arange(0,1,delta)
        delta_ecdf = np.abs(ecdf0(probing_points) - ecdf1(probing_points))

        indices = np.arange(2*K).reshape((-1,1)) + np.arange(1,np.ceil(1/delta)-2*K+1).astype(int)
        mcdp = max(mcdp, np.max(np.min(delta_ecdf[indices],axis=0)))*100
        mcdp_list.append(mcdp)
    return mcdp_list



def demographic_parity(y_pred: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the demographic parity, which measures the difference in positive rate between different
    values of a binary sensitive attribute. The positive rate is defined as the proportion of data points
    that are predicted to be positive.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in positive rate
        between different values of the sensitive attribute.
    """
    # Convert predicted probabilities to binary predictions using the threshold.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate the difference in positive rate.
    parity = abs(y_z_1.mean() - y_z_0.mean())
    parity *= 100
    return parity



def metric_evaluation(y_gt, y_pre, s, s_pre=None, prefix="", epsilon_list=[0.01,0.05,0.1], K=32):
    """
    Calculate various performance and fairness metrics based on ground truth labels, predicted values, and sensitive attributes.

    Parameters:
    y_gt (numpy.ndarray): A one-dimensional array of ground truth labels, where each label is either 0 or 1.
    y_pre (numpy.ndarray): A one-dimensional array of predicted values between 0 and 1.
    s (numpy.ndarray): A one-dimensional array of sensitive attributes, where each attribute is either 0 or 1.
    s_pre (numpy.ndarray): A one-dimensional array of predicted sensitive attributes, where each attribute is either 0 or 1. Default is None.
    prefix (str): A string to prefix all metric names with. Default is an empty string.

    Returns:
    dict: A dictionary that maps metric names to values.
    """

    # Flatten the input arrays
    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    ap = metrics.average_precision_score(y_gt, y_pre) * 100
    dp = demographic_parity(y_pre, s, threshold=0.5)
    dpe = demographic_parity(y_pre, s, threshold=None)
    abcc = ABCC(y_pre, y_gt, s)
    mcdp0 = MCDP_exact(y_pre, y_gt, s, epsilon=0)
    mcdp_a = MCDP_approximate(y_pre, y_gt, s, epsilon_list, K)
    # mcdp_a = []
    # for e in epsilon_list:
    #     mcdp_a.append(MCDP_exact(y_pre, y_gt, s, e, K))

    metric_name = ["ap", "dp", "dpe", "abcc", "mcdp_a", "mcdp0"]
    metric_name = [prefix + "/" + x for x in metric_name]
    metric_val = [ap, dp, dpe, abcc, mcdp_a, mcdp0]

    return dict(zip(metric_name, metric_val))
