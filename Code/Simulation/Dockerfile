FROM python:3.11-slim

RUN pip install pyzmq

WORKDIR /app
COPY src/lobe.py .

CMD ["python", "lobe.py"]
