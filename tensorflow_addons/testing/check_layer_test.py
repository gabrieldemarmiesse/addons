from tensorflow.keras.layers import Dense
from tensorflow_addons.testing.check_layer import check_layer
import numpy as np


def test_check_layer():
    check_layer(
        Dense,
        {"units": 2},
        input_shape=(None, 6),
        input_data=np.random.uniform(0, 1, (4, 6)).astype(np.float32),
    )
