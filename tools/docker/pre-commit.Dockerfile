FROM python:3.6

COPY tools/tests_dependencies /tests_dependencies
RUN pip install -r /tests_dependencies/black.txt -r /tests_dependencies/flake8.txt

COPY tools/ci_build/install/buildifier.sh ./buildifier.sh
RUN bash buildifier.sh

COPY tools/ci_build/install/clang-format.sh ./clang-format.sh
RUN bash clang-format.sh

RUN pip install git+https://github.com/timothycrosley/isort.git@a04700812bd4b9eb687065d1199b52bb8e045774

# hack to make for isort to recognize addons as first party
RUN cd /tmp && \
    git clone https://github.com/tensorflow/addons.git && \
    pip install -e ./addons

WORKDIR /addons


CMD ["python", "tools/format.py"]
