FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

COPY ../stock-rnn-prediction.py /opt/stock-rnn-prediction/
COPY ./requirements.txt /opt/stock-rnn-prediction/

RUN pip3 install -r /opt/stock-rnn-prediction/requirements.txt

ENTRYPOINT ["python3", "/opt/stock-rnn-prediction/stock-rnn-prediction.py"]
