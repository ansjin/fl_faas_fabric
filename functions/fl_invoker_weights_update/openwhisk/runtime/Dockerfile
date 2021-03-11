FROM openwhisk/python3action
COPY requirements.txt requirements.txt

RUN apk update \
    && apk add --virtual build-deps gcc python3-dev musl-dev \
    && apk del build-deps

RUN  pip install --no-cache-dir -r requirements.txt