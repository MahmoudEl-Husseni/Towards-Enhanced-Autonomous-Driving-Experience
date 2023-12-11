FROM tensorflow/tensorflow
ADD . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r tf_requirments.txt
ENV PYTHONUNBUFFERED=1
