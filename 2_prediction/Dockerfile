FROM tensorflow/tensorflow:latest
RUN apt-get -y update && apt-get install -y --no-install-recommends \
		 apt-utils \
		 ca-certificates \
		 curl \
		 git \
		 rsync \
		 nginx \
		 software-properties-common \
		 vim \
         wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install \
        boto3 \
        keras \
		matplotlib \
        pandas \
        pytz \
        pyyaml \
        s3fs \
		sklearn

RUN pip --no-cache-dir install \
        flask \
        gevent \
        gunicorn
		
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

RUN mkdir -p /opt/ml/code
RUN mkdir -p /opt/ml/failure
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/output

COPY model /opt/ml/code
RUN chmod -R 777 /opt/ml/code
WORKDIR /opt/ml/code