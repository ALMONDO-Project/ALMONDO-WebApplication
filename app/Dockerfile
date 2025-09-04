FROM python:3.11.4-alpine3.18

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["gunicorn", "--config", "Gunicorn/gunicorn_config.py", "--log-level=info", "--access-logfile", "-", "app:app"]
