FROM python:3.7

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV HOME=/app
COPY requirements.txt ${HOME}/
RUN pip install -r ${HOME}/requirements.txt

WORKDIR ${HOME}

COPY . ${HOME}/

# Install extra datasets
RUN wget https://download.pytorch.org/tutorial/hymenoptera_data.zip && unzip hymenoptera_data.zip
RUN git clone https://github.com/tjmoon0104/pytorch-tiny-imagenet.git && cd pytorch-tiny-imagenet && ./run.sh

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]