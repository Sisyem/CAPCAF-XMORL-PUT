 # Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run main.py
CMD ["python", "main.py"]
