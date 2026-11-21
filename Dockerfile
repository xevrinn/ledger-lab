FROM python:3.11-slim
WORKDIR /app
COPY . .
ENV AUDIT_HOST=0.0.0.0 AUDIT_PORT=8000
EXPOSE 8000
CMD ["python", "server.py"]
