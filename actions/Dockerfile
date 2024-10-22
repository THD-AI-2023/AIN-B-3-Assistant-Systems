FROM rasa/rasa-sdk:3.6.2

WORKDIR /app

# Copy actions code first to leverage Docker cache
COPY . /app

USER root

# Install build tools and Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements-actions.txt

# (Optional) Install specific spacy models if required
# RUN python -m spacy download en_core_web_sm

USER 1001

CMD ["start", "--actions", "actions"]
