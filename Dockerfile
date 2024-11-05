FROM python:3.9

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code
COPY src/ ./src/

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app with explicit server address and port
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
