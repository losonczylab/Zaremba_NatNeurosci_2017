# To build (from root directory of git repo):
#   docker build -t zaremba_nn_2017 -f docker/Dockerfile .

FROM losonczylab/zaremba_nn_2017_data

LABEL maintainer="jzaremba@gmail.com"

RUN apt-get update -qq && apt-get install -qqy \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN mkdir /code
COPY . /code
RUN cd /code/losonczy_analysis_bundle && \
    python setup.py install
COPY docker/docker_paths.cfg /code/paths.cfg

VOLUME /analysis

WORKDIR /code

COPY docker/entrypoint.sh /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["python", "run_scripts.py"]
