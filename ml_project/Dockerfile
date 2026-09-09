FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN pip install --no-cache-dir -e /app

# Default command runs a training job
CMD ["python", "-m", "ml_project.train", "--n-samples", "500"]