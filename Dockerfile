FROM python:3.6

RUN apt-get update && apt-get -y install python3-tk

COPY . .
RUN pip install -r requirements.txt

CMD [ "python3", "spam-classifier.py", "-d", "dataset/spambase.arff", "-c", "all", "--verbose", "-s"]