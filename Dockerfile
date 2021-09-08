FROM nvcr.io/nvidia/pytorch:21.08-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update --fix-missing && \
    apt-get install --yes --no-install-recommends \
         tzdata dialog apt-utils \ 
         gcc pkg-config git curl build-essential \
         ffmpeg libsm6 libxext6 libpcre3 libpcre3-dev

RUN useradd --gid root --home-dir /usr/local/descriptor --create-home descriptor  
WORKDIR /usr/local/descriptor
COPY . ./

ENV VIRTUAL_ENV=/opt/venv
RUN chmod -R g+rwx /usr/local/descriptor && \
    python -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip && pip install loguru 

ENV SOURCE="source/"
ENV NB_WORKERS=2
ENV PUBLISHER_PORT=7600
ENV BACK_ROUTER_PORT=8101
ENV FRONT_ROUTER_PORT=8100

EXPOSE $PUBLISHER_PORT
EXPOSE $BACK_ROUTER_PORT
EXPOSE $FRONT_ROUTER_PORT 

CMD ["python", "main.py"]