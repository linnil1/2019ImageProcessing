FROM python:3.7-stretch
RUN pip3 install numpy matplotlib PyQtChart opencv-python && \
    apt update -y && \
    apt install -y python3-pyqt5

COPY . /app/
WORKDIR /app/
RUN python3 setup.py install
