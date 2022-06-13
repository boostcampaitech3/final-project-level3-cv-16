FROM python:3.9-slim-buster

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONBUFFERED=1
ENV MYSQL_USER="root"
ENV MYSQL_PASSWORD="1234"
ENV MYSQL_HOST="104.154.196.9"

RUN apt-get update ; apt-get upgrade -y

RUN apt-get install libgl1-mesa-glx -y

RUN apt-get install libglib2.0-0 -y
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python","backend.py"]