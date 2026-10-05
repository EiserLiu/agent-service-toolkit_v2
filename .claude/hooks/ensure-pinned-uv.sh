#!/bin/bash
# 将沙箱中的 uv 固定为仓库指定版本。pyproject.toml 中相对时间形式的
# `exclude-newer` 冷却期需要较新的 uv；旧版本在重新生成锁文件时，
# 会静默删除 uv.lock 中的冷却期元数据（`uv run` / `uv sync` 也可能
# 触发此操作）。使用与 Dockerfile 相同的 `pip install uv==` 方式
# 从 PyPI 安装，并从 Dockerfile 获取版本号，避免额外维护一份
# 版本配置。仅修改托管远程沙箱中的 uv，
# 不会修改本地开发者安装的 uv。
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

project_dir="${CLAUDE_PROJECT_DIR:-.}"
pinned="$(grep -hoE 'uv==[0-9]+\.[0-9]+\.[0-9]+' "$project_dir"/docker/Dockerfile.* 2>/dev/null | head -1 | cut -d= -f3)"
[ -n "$pinned" ] || exit 0

current="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
[ "$current" = "$pinned" ] && exit 0

python3 -m pip install --user --quiet --no-cache-dir "uv==$pinned" >/dev/null 2>&1 || true
exit 0
