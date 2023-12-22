
# FROM python:3.11-slim
# from official python repo, grab 3.11-alpine
FROM python:3.11-alpine

RUN apk add build-base linux-headers  # # alpine misses this but its reqiured to build psutil


WORKDIR /app

COPY requirements.txt requirements.txt
# copied separately at the top in order to cache it even though code might change
RUN pip install --upgrade pip
RUN pip install -v -r requirements.txt

COPY . code/crocodile
RUN pip install --no-cache-dir -e code/crocodile

# [full]
# docker has its own cache dir, we don't want it,
# CMD ["python", "code/crocodile/my_project/manage.py", "runserver"]
# COPY $HOME/code/machineconfig code/machineconfig
