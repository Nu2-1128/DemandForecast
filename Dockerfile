# Use a Python slim image for a smaller final image size
FROM python:3.10-slim

# Expose the port that Streamlit will run on
EXPOSE 8080

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to take advantage of Docker's layer caching.
# Make sure your requirements.txt file now includes cyipopt
COPY requirements.txt .

# Install all Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container.
COPY . .

# Command to run your Streamlit application
ENTRYPOINT ["streamlit", "run", "DemandForecast.app.py", "--server.port=8080","--server.address=0.0.0.0"]