FROM nvcr.io/nvidia/pytorch:23.08-py3
ADD . .
RUN cd rasterize/src
RUN pip install -r requirements.txt
CMD ["python3", "main.py"]