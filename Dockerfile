# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and source code
COPY requirements.txt ./
COPY algo_trading.py ./
COPY README.md ./
# If you want to use .env and service account JSON, copy them too (or mount as secrets in Render)
# COPY .env ./
# COPY htoh-468014-c8802914cedc.json ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (optional, for Render secrets)
# ENV BOT_TOKEN=your_token
# ENV CHAT_ID=your_chat_id

# Default command: run main script
CMD ["python", "algo_trading.py"]
