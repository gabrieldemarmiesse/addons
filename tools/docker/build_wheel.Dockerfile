ARG PY_VERSION

FROM tensorflow/tensorflow:2.1.0-custom-op-gpu-ubuntu16 as make_wheel

COPY tools/docker/finish_bazel_install.sh .
RUN bash finish_bazel_install.sh
RUN apt-get update && apt-get install patchelf

ARG PY_VERSION
RUN python$PY_VERSION -m pip install --upgrade pip setuptools auditwheel==2.0.0

COPY tools/install_deps/ /install_deps
RUN python$PY_VERSION -m pip install \
        -r /install_deps/tensorflow.txt \
        -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python$PY_VERSION -m pip install -r requirements.txt

COPY ./ /addons
WORKDIR /addons
ARG NIGHTLY_FLAG
RUN bash tools/releases/release_linux.sh $PY_VERSION $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

FROM python:$PY_VERSION as test_in_fresh_environment

COPY tools/install_deps/tensorflow.txt .
RUN pip install -r tensorflow.txt

COPY --from=make_wheel /addons/wheelhouse/ /wheelhouse
RUN pip install /wheelhouse/*.whl

RUN python -c "import tensorflow_addons as tfa; print(tfa.activations.lisht(0.2))"

FROM scratch as output

# to avoid triggering dead branch elimination, we
# need to use the second stage. Otherwise it's going
# to be removed by docker buildkit.
COPY --from=test_in_fresh_environment /wheelhouse/ .
