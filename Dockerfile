FROM python:3.9 as builder
MAINTAINER Peter Michael Schwarz "peter.schwarz@uni-marburg.de"

COPY . /DR4DNA
WORKDIR /DR4DNA

RUN apt-get update -y \
 && apt-get install --no-install-recommends -y python-dev software-properties-common gcc virtualenv build-essential cmake make\
 && python -m pip install numpy \
 && python -m pip install pandas

# setup NOREC4DNA + dependencies
WORKDIR /DR4DNA/NOREC4DNA
RUN find /DR4DNA/NOREC4DNA -name '*.pyc' -delete
RUN rm -rf /DR4DNA/NOREC4DNA/venv && python -m venv venv && . /DR4DNA/NOREC4DNA/venv/bin/activate && pip install wheel && pip install -r requirements.txt && python setup.py install

WORKDIR /DNA_Aeon

RUN pip install -r requirements.txt && python setup.py install

RUN apt-get purge -y --auto-remove build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# squash / reduce size
FROM scratch
COPY --from=builder / /

WORKDIR /DR4DNA
ENTRYPOINT ["python", "app.py"]
#ENTRYPOINT ["bash"]
#CMD ["bash"]