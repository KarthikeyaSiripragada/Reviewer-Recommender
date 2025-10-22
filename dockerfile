# Dockerfile for Hugging Face Space (Streamlit)
FROM python:3.10-slim

# set a safe working directory
WORKDIR /app

# copy all project files into container
COPY . /app

# install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---- FIX: Streamlit needs a writable config directory ----
ENV HOME=/app
RUN mkdir -p /app/.streamlit && chmod -R 777 /app/.streamlit
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
# ----------------------------------------------------------

# expose Streamlit's default port
EXPOSE 7860

# run the Streamlit app
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
