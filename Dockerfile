FROM python:3.9

WORKDIR /mle-training

RUN apt-get update && apt-get install -y git

COPY . .

RUN pip install build
RUN python -m build
RUN pip install mlflow


RUN pip install -r requirements.txt

CMD ["sh", "-c", "python ./src/housing/run_script.py && pytest && mlflow ui"]

