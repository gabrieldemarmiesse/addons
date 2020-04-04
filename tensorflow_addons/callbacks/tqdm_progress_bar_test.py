from distutils.version import LooseVersion
import numpy as np
import pytest
import tensorflow as tf

import tensorflow_addons as tfa


def get_data_and_model():
    x = np.random.random((5, 1))
    y = np.random.randint(0, 2, (5, 1), dtype=np.int)

    inputs = tf.keras.layers.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="sgd", loss="mse", metrics=["acc"])
    return x, y, model


@pytest.mark.xfail(
    LooseVersion(tf.__version__) >= LooseVersion("2.2.0"),
    reason="TODO: Fixeme See #1495",
)
def test_tqdm_progress_bar(capsys):

    x, y, model = get_data_and_model()

    capsys.readouterr()  # flush the buffer
    model.fit(x, y, epochs=1, verbose=0, callbacks=[tfa.callbacks.TQDMProgressBar()])
    fit_stderr = capsys.readouterr().err
    assert "loss:" in fit_stderr
    assert "acc:" in fit_stderr
