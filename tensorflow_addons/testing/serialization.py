from typing import Union, List, Optional
import inspect

import numpy as np
from tensorflow.keras.metrics import Metric
from tensorflow.keras.layers import Layer
import typeguard

ShapeOrArray = Union[tuple, np.ndarray]


@typeguard.typechecked
def check_metric_serialization(
    metric: Metric,
    y_true: ShapeOrArray,
    y_pred: ShapeOrArray,
    sample_weight: Optional[ShapeOrArray] = None,
    strict: bool = True,
):
    config = metric.get_config()
    class_ = metric.__class__

    check_config(config, class_, strict)

    metric_copy = class_(**config)
    metric_copy.set_weights(metric.get_weights())

    if isinstance(y_true, tuple):
        y_true = get_random_array(y_true)
    if isinstance(y_pred, tuple):
        y_pred = get_random_array(y_pred)
    if isinstance(sample_weight, tuple) and sample_weight is not None:
        sample_weight = get_random_array(sample_weight)

    # the behavior should be the same for the original and the copy
    if sample_weight is None:
        metric.update_state(y_true, y_pred)
        metric_copy.update_state(y_true, y_pred)
    else:
        metric.update_state(y_true, y_pred, sample_weight)
        metric_copy.update_state(y_true, y_pred, sample_weight)

    assert_all_arrays_close(metric.get_weights(), metric_copy.get_weights())
    metric_result = metric.result().numpy()
    metric_copy_result = metric_copy.result().numpy()
    if metric_result != metric_copy_result:
        raise ValueError(
            "The original gave a result of {} after an "
            "`.update_states()` call, but the copy gave "
            "a result of {} after the same "
            "call.".format(metric_result, metric_copy_result)
        )


@typeguard.typechecked
def check_layer_serialization(
    layer: Layer,
    input_data: Union[ShapeOrArray, List[ShapeOrArray]],
    strict: bool = True,
):
    if isinstance(input_data, tuple):
        input_data = get_random_array(input_data)
    if isinstance(input_data, list) and isinstance(input_data[0], tuple):
        input_data = [get_random_array(x) for x in input_data]
    if isinstance(input_data, list):
        input_shape = [x.shape for x in input_data]
    else:
        input_shape = input_data.shape

    layer.build(input_shape)
    config = layer.get_config()
    class_ = layer.__class__

    check_config(config, class_, strict)

    layer_copy = class_.from_config(config)
    layer_copy.build(input_shape)
    layer_copy.set_weights(layer.get_weights())

    assert layer.name == layer_copy.name
    assert layer.dtype == layer_copy.dtype

    output_data = [x.numpy() for x in to_list(layer(input_data))]
    output_data_copy = [x.numpy() for x in to_list(layer_copy(input_data))]
    assert_all_arrays_close(output_data, output_data_copy)


def check_config(config, class_, strict):
    init_signature = inspect.signature(class_.__init__)

    for parameter_name in init_signature.parameters:
        if parameter_name == "self":
            continue
        elif parameter_name == "args":
            if strict:
                raise KeyError(
                    "Please do not use args in the class constructor of {}, "
                    "as it hides the real signature "
                    "and degrades the user experience. "
                    "If you have no alternative to *args, "
                    "use `strict=False` in check_metric_serialization.".format(
                        class_.__name__
                    )
                )
            else:
                continue
        elif parameter_name == "kwargs":
            if strict:
                raise KeyError(
                    "Please do not use kwargs in the class constructor of {}, "
                    "as it hides the real signature "
                    "and degrades the user experience. "
                    "If you have no alternative to **kwargs, "
                    "use `strict=False` in check_metric_serialization.".format(
                        class_.__name__
                    )
                )
            else:
                continue
        if parameter_name not in config:
            raise KeyError(
                "The constructor parameter {} is not present in the config dict "
                "obtained with `.get_config()` of {}. All parameters should be set to "
                "ensure a perfect copy of the keras object can be obtained when "
                "serialized.".format(parameter_name, class_.__name__)
            )


def to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def assert_all_arrays_close(list1, list2):
    for array1, array2 in zip(list1, list2):
        np.testing.assert_allclose(array1, array2)


def get_random_array(shape):
    return np.random.uniform(size=shape).astype(np.float32)
