# To build (from root directory of git repo):
#   docker build -t zaremba_nn_2017 -f docker/Dockerfile .

FROM losonczylab/zaremba_nn_2017_data

LABEL maintainer="jzaremba@gmail.com"

RUN apt-get update -qq && apt-get install -qqy \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# There is a bug that arises with pandas >0.19.2,
# (tested 0.20.0, 0.20.3, 0.21.0rc1)
# To avoid this, I will pin pandas to 0.19.2 here.
RUN pip install -U --no-deps pandas==0.19.2

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
