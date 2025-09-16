FROM python:3.12-slim
ARG ARCH=amd64
ARG SSH_PRIVATE_KEY=""
ARG VERSION="N/A"
ENV VERSION=$VERSION
ENV SSH_PRIVATE_KEY=$SSH_PRIVATE_KEY

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN DEBIAN_FRONTEND=noninteractive apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    openssh-server \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN mkdir -p ~/.ssh \
 && echo "StrictHostKeyChecking no" >> ~/.ssh/config \
 && echo "UserKnownHostsFile=/dev/null" >> ~/.ssh/config \
 && echo "$SSH_PRIVATE_KEY" > ~/.ssh/id_ed25519 \
 && chmod 600 ~/.ssh/id_ed25519 \
 && uv sync --no-cache --extra ml\
 && rm -rf ~/.ssh

CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--root-path", "/", "--port", "80", "--host", "0.0.0.0"]