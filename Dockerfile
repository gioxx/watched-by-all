FROM python:3.14-slim

ARG APP_VERSION=dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV APP_VERSION=${APP_VERSION}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# Bake the build-time version into static/app_meta.json so the webapp
# footer and inline defaults always match the released tag. When
# building outside a tag (APP_VERSION=dev) the file is left untouched.
RUN if [ "$APP_VERSION" != "dev" ] && [ -f static/app_meta.json ]; then \
        python -c "import json,os; p='static/app_meta.json'; d=json.load(open(p)); d['version']=os.environ['APP_VERSION']; json.dump(d, open(p,'w'), indent=2)"; \
    fi

EXPOSE 8088

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
