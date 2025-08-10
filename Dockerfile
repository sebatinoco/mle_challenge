# Define the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Use CMD to run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]