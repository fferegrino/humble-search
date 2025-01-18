FROM python:3.11-slim

WORKDIR /app

COPY ./containers/data-load/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MODEL_PATH=/app/model.bin

COPY ./containers/data-load/model_setup.py model_setup.py

RUN python model_setup.py

COPY ./containers/data-load/*.py ./

CMD ["python", "setup.py"]
