FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
USER root
CMD ["python3"]