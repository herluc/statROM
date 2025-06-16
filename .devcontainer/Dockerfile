#FROM dolfinx/dolfinx:nightly
FROM dolfinx/dolfinx:v0.6.0

COPY requirements.txt ./


RUN apt-get update && \
      apt-get -y install sudo

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir --upgrade matplotlib \
  && pip install --no-cache-dir -r requirements.txt \
  && sudo apt-get update \
  && sudo apt-get --yes install texlive \
  && sudo apt-get --yes install texlive-latex-extra texlive-fonts-recommended dvipng \
  && sudo apt-get --yes install cm-super


COPY . .
