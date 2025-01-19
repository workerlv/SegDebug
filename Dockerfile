# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8512