FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.9
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
CMD ["python3"]