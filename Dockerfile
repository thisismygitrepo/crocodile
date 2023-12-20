
FROM python:3.11-slim


WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -v -r requirements.txt

COPY . code/crocodile
RUN pip install --no-cache-dir -e code/crocodile  
# [full]
# docker has its own cache dir, we don't want it,
# CMD ["python", "code/crocodile/my_project/manage.py", "runserver"]
# COPY $HOME/code/machineconfig code/machineconfig
