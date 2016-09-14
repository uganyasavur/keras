import numpy as np
from . import backend as K


def mean_squared_prediction_error(y_true, y_pred):
    return K.mean(K.square(y_true - K.round(y_pred)))


def mean_squared_prediction_error_ignore(y_true, y_pred):
    spe = K.square(y_true - K.round(y_pred))
    mask = K.not_equal(y_true, -12345)
    a = K.sum(mask)
    a = a + K.equal(a, 0)
    mspe_i = K.sum(spe*mask)/a
    return mspe_i


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def binary_accuracy_ignore(y_true, y_pred):
    acc = K.equal(y_true, K.round(y_pred))
    mask = K.not_equal(y_true, -12345)
    a = K.sum(mask)
    a = a + K.equal(a, 0)
    acc_i = K.sum(acc*mask)/a
    return acc_i

def binary_accuracy_ignore0(y_true, y_pred):
    acc = K.equal(y_true, K.round(y_pred))
    mask = K.not_equal(y_true, 0)
    a = K.sum(mask)
    a = a + K.equal(a, 0)
    acc_i = K.sum(acc*mask)/a
    return acc_i


def categorical_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))


def sparse_categorical_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def mean_squared_error_ignore(y_true, y_pred):
    se = K.square(y_true - K.round(y_pred))
    mask = K.not_equal(y_true, -12345)
    a = K.sum(mask)
    a = a + K.equal(a, 0)
    mse_i = K.sum(se*mask)/a
    return mse_i


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log))


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)))


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.))


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.mean(K.categorical_crossentropy(y_pred, y_true))


def sparse_categorical_crossentropy(y_true, y_pred):
    '''expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    '''
    return K.mean(K.sparse_categorical_crossentropy(y_pred, y_true))


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))


def binary_crossentropy_ignore(y_true, y_pred):
    bc = K.binary_crossentropy(y_pred, y_true)
    mask = K.not_equal(y_true, -12345)
    a = K.sum(mask)
    a = a + K.equal(a, 0)
    bc_i = K.sum(bc*mask)/a
    return bc_i

def binary_crossentropy_ignore0(y_true, y_pred):
    bc = K.binary_crossentropy(y_pred, y_true)
    mask = K.not_equal(y_true, 0)
    a = K.sum(mask)
    a = a + K.equal(a, 0)
    bc_i = K.sum(bc*mask)/a
    return bc_i


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred)


def matthews_correlation(y_true, y_pred):
    ''' Matthews correlation coefficient
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(1 - y_neg * y_pred_pos)
    fn = K.sum(1 - y_pos * y_pred_neg)
    
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# aliases
mse = MSE = mean_squared_error
mspe = MSPE = mean_squared_prediction_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'metric')
