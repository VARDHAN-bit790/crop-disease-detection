# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 5000

# Command to run your app
CMD ["python", "app.py"]
