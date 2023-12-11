FROM nvcr.io/nvidia/pytorch:23.08-py3
ADD . .
RUN pip3 install -r torch_requirments.txt
ENV PYTHONUNBUFFERED=1
CMD ["python3", "/workspace/train.py"]
