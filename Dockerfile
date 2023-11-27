FROM python:3.9 AS builder
MAINTAINER Peter Michael Schwarz "peter.schwarz@uni-marburg.de"

COPY . /DR4DNA
WORKDIR /DR4DNA

RUN apt-get update -y \
 && apt-get install --no-install-recommends -y software-properties-common gcc virtualenv build-essential cmake make

# setup NOREC4DNA + dependencies
WORKDIR /DR4DNA/NOREC4DNA
RUN find /DR4DNA/NOREC4DNA -name '*.pyc' -delete
RUN rm -rf /DR4DNA/NOREC4DNA/venv &&  pip install wheel && pip install -r requirements.txt && python setup.py install

WORKDIR /DR4DNA
RUN pip install -r requirements.txt && python setup.py install

RUN apt-get purge -y --auto-remove build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# squash / reduce size
FROM scratch
COPY --from=builder / /

WORKDIR /DR4DNA/working_dir

ENTRYPOINT ["python", "../app.py", "input.ini"]