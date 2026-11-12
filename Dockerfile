# docker build -t atlas . && docker run -p 8010:8010 -v "$PWD:/books" -v atlas-home:/root/.atlas atlas sh -c "atlas import /books/mybooks.csv && atlas serve --host 0.0.0.0 --no-browser"
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 ATLAS_HOST=0.0.0.0 OLLAMA_HOST=http://host.docker.internal:11434
WORKDIR /app
COPY . /app
RUN pip install --pre . && rm -rf /root/.cache
EXPOSE 8010
CMD ["atlas", "serve", "--host", "0.0.0.0", "--no-browser"]
