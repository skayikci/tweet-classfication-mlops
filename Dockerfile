FROM python:3.10-slim-bookworm

# Avoid interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install security patches
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Optional: install curl if needed for debugging
# RUN apt-get install -y curl

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port (if needed)
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
