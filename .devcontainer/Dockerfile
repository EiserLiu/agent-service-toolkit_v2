FROM python:3.13.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 vscode

RUN pip install --no-cache-dir uv==0.12.5

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

RUN mkdir -p /opt/venv /home/vscode/.cache/uv /workspaces/agent-service-toolkit \
    && chown -R vscode:vscode /opt/venv /home/vscode/.cache /workspaces

WORKDIR /workspaces/agent-service-toolkit
COPY --chown=vscode:vscode pyproject.toml uv.lock ./
USER vscode
RUN uv sync --frozen --no-install-project

CMD ["sleep", "infinity"]
