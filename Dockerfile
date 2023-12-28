FROM python:3.11.5

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5051

CMD ["gunicorn", "-b", "0.0.0.0:5051", "app:app"]