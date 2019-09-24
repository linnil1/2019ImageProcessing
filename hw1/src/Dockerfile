FROM python:3.7-stretch
RUN pip3 install numpy matplotlib PyQtChart && \
    apt update -y && \
    apt install -y python3-pyqt5

COPY . /app/src/
WORKDIR /app/src
RUN python3 setup.py install
