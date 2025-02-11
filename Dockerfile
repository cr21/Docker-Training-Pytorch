FROM python:3.9.19-slim AS stg1

    
COPY requirements.txt .

RUN apt-get update -y && apt install -y --no-install-recommends git\
&& pip install --no-cache-dir -U pip \ 
    && pip install --user --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    && pip install --user --no-cache-dir https://download.pytorch.org/whl/cpu/torchvision-0.12.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    && pip install --user --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Stage 2: run application code
FROM python:3.9.19-slim

COPY --from=stg1 /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /src

COPY . .

# ENTRYPOINT ["python3", "inference.py"]
