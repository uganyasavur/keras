import numpy as np
import pytest

from keras.models import Sequential
from keras.layers.core import Masking
from keras.layers import Dense
from keras.layers.wrappers import TimeDistributed
from keras.utils.test_utils import keras_test
from keras.engine.training import masked_metric
from keras import metrics
from keras import backend as K

@keras_test
def test_masking():
    np.random.seed(1337)
    X = np.array([[[1], [1]],
                  [[0], [0]]])
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(2, 1)))
    model.add(TimeDistributed(Dense(1, init='one')))
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
    y = np.array([[[1], [1]],
                  [[1], [1]]])
    metric = model.evaluate(X, y)[1]
    assert metric == 0

@keras_test
def test_masking_binary_accuracy():
    masking_function_test('binary_accuracy')

@keras_test
def test_masking_mse():
    masking_function_test('mse')

@keras_test
def test_masking_mae():
    masking_function_test('mae')

@keras_test
def test_masking_mape():
    masking_function_test('mape')

@keras_test
def test_masking_msle():
    masking_function_test('msle')

@keras_test
def test_masking_hinge():
    masking_function_test('hinge')

@keras_test
def test_masking_squared_hinge():
    masking_function_test('squared_hinge')

@keras_test
def test_masking_poisson():
    masking_function_test('poisson')



def masking_function_test(fn_name):
    fn = metrics.get(fn_name)
    fn_masked = masked_metric(fn)
    shape = (3, 4, 2)
    X1 = np.arange(24).reshape(shape)
    X2 = np.concatenate((X1, np.zeros(6).reshape(3,1,2)), axis=1)

    Y1 = 2*X1
    Y2 = 2*X2

    mask = np.ones((3,5))
    mask[:,-1] = 0

    mse_X1_Y1         = K.eval(fn(K.variable(X1),
                            K.variable(Y1)))

    mse_X1_Y1_mask_fn = K.eval(fn_masked(K.variable(X1),
                            K.variable(Y1)))

    mse_X2_Y2         = K.eval(fn(K.variable(X2),
                            K.variable(Y2)))

    mse_X2_Y2_mask_fn = K.eval(fn_masked(K.variable(X2),
                                   K.variable(Y2)))

    mse_X2_Y2_masked  = K.eval(fn_masked(K.variable(X2),
                                   K.variable(Y2),
                                   K.variable(mask)))

    # without mask metric output should be independent of
    # which function is used
    # use almost equal to account for float precision
    assert abs(mse_X1_Y1 - mse_X1_Y1_mask_fn) < 0.0001, "Masked value not computed correctly for metric %s" % fn
    assert abs(mse_X2_Y2 - mse_X2_Y2_mask_fn) < 0.0001, "Masked value not computed correctly for metric %s" % fn

    # masked mse X2-Y2 should be equal to mse X1-Y1
    # use almost equal to account for float precision
    assert abs(mse_X1_Y1 - mse_X2_Y2_masked) < 0.0001, "Masked value not computed correctly for metric %s" % fn


if __name__ == '__main__':
    pytest.main([__file__])
