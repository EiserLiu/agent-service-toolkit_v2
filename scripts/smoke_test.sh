#!/usr/bin/env bash
# 按需测试未纳入快速单元测试和默认 CI 的 Docker 集成链路：
# PostgreSQL 和 MongoDB 检查点保存器、
# AG-UI 接口及 Langfuse 追踪。维护者或 Agent
# 无需等待完整 CI 周期即可验证这些功能。
#
# 用法：
#   ./scripts/smoke_test.sh                 # 默认目标：postgres、mongo、agui
#   ./scripts/smoke_test.sh mongo           # 运行单个目标
#   ./scripts/smoke_test.sh postgres agui   # 运行部分目标
#   ./scripts/smoke_test.sh langfuse        # 单独运行较重的 langfuse 目标
#   ./scripts/smoke_test.sh all             # 运行全部目标，包括 langfuse
#
# 目标：postgres、mongo、agui、langfuse
# 默认不运行 langfuse，因为它会启动完整的 Langfuse
# 自托管服务栈（6 个服务，约 5GB 镜像），耗时明显更长。
# 请显式指定或通过 `all` 运行。还需能访问 cgr.dev 容器镜像仓库
# 以拉取 minio 镜像；在限制出站访问的云环境中，
# 请先将 cgr.dev 加入网络允许列表。
#
# 只有数据库在 Docker 中运行；服务本身通过 uv 在宿主机运行，
# 并连接 localhost。这是有意设计的：构建服务镜像时需要
# 从构建容器内访问包仓库，而某些 Agent 沙箱环境
# 会阻止此类访问。宿主机运行可避开此限制，
# 同时仍能验证真实数据库容器。
#
# 测试通过应有实际依据。除 pytest/API 检查外，
# 各目标还会独立确认确实使用了预期依赖：
# 数据库目标直接查询容器中本次运行的会话；langfuse
# 通过 API 查询追踪记录。这能发现静默回退（如回退到 SQLite），
# 否则任何可用检查点后端都可能让 API 层测试误通过。
#
# 依赖：docker、docker compose、uv、node（AG-UI 客户端）、python3、curl
set -euo pipefail
cd "$(dirname "$0")/.."

# 每次运行使用唯一标识，即使数据库卷非空，后续验证
# 也只检查本次运行的数据。导出此变量，让 pytest 使用相同的
# 会话 ID（见 tests/smoke/test_persistence.py）。
SMOKE_THREAD_ID="smoke-test-$(date +%s)-$$"
export SMOKE_THREAD_ID

# 从上游指定标签获取 Langfuse 自托管 Compose 配置，不将其副本纳入仓库。
# 升级此标签即可使用新版 Langfuse。
LANGFUSE_REF="v3.225.5"

SERVICE_PID=""
SERVICE_LOG=""
LANGFUSE_COMPOSE=""  # 临时 Compose 文件，在运行 langfuse 目标时设置

start_service() {
  # 使用指定后端环境在宿主机启动 Agent 服务，然后等待
  # 健康检查通过。参数：KEY=VALUE ...，用于指定连接设置。
  if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
    echo "  ✗ refusing to start: something is already listening on :8080"
    return 1
  fi
  SERVICE_LOG="$(mktemp)"
  env USE_FAKE_MODEL=true "$@" uv run python src/run_service.py > "$SERVICE_LOG" 2>&1 &
  SERVICE_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      echo "  ✗ service exited during startup; log:"
      cat "$SERVICE_LOG"
      return 1
    fi
    sleep 2
  done
  echo "  ✗ service did not become healthy within 60s; log:"
  cat "$SERVICE_LOG"
  return 1
}

stop_service() {
  if [[ -n "$SERVICE_PID" ]]; then
    kill "$SERVICE_PID" 2>/dev/null || true
    wait "$SERVICE_PID" 2>/dev/null || true
    SERVICE_PID=""
  fi
  if [[ -n "$SERVICE_LOG" ]]; then
    rm -f "$SERVICE_LOG"
    SERVICE_LOG=""
  fi
}

wait_healthy() {
  # 等待容器健康检查通过。参数：容器 ID。
  local cid="$1" status=""
  for _ in $(seq 1 20); do
    status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo missing)"
    [[ "$status" == "healthy" ]] && return 0
    echo "  waiting for database... ($status)"
    sleep 2
  done
  echo "  ✗ database never became healthy"
  return 1
}

assert_positive_count() {
  # 参数：计数字符串、标签。计数不是大于 0 的整数时明确报错。
  local n="$1" label="$2"
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )); then
    echo "  ✓ verified: $n $label"
  else
    echo "  ✗ FAIL: expected >0 $label, got '$n' — backend was NOT exercised as intended"
    return 1
  fi
}

cleanup() {
  echo "--- Tearing down ---"
  stop_service
  # down 会移除合并项目中的所有服务（postgres + mongo），
  # 因此无论之前运行哪个目标，一次调用即可完成清理。
  docker compose -f compose.yaml -f docker/compose.mongo.yaml down -v >/dev/null 2>&1 || true
  if [[ -n "$LANGFUSE_COMPOSE" && -f "$LANGFUSE_COMPOSE" ]]; then
    docker compose -f "$LANGFUSE_COMPOSE" down -v >/dev/null 2>&1 || true
    rm -f "$LANGFUSE_COMPOSE"
  fi
}
trap cleanup EXIT

smoke_postgres() {
  echo "=== Postgres checkpointer (DATABASE_TYPE=postgres) ==="
  docker compose -f compose.yaml up -d postgres
  local cid
  cid="$(docker compose -f compose.yaml ps -q postgres)"
  wait_healthy "$cid"
  start_service DATABASE_TYPE=postgres POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  local n
  n="$(docker exec -e PGPASSWORD=postgres "$cid" psql -U postgres -d agent_service -tAc \
    "select count(*) from checkpoints where thread_id='$SMOKE_THREAD_ID'" 2>/dev/null | tr -d '[:space:]')" || true
  assert_positive_count "$n" "postgres checkpoint rows for this run's thread"
  stop_service
  docker compose -f compose.yaml down -v
}

smoke_mongo() {
  echo "=== MongoDB checkpointer (DATABASE_TYPE=mongo) ==="
  local files=(-f compose.yaml -f docker/compose.mongo.yaml)
  docker compose "${files[@]}" up -d mongo
  local cid
  cid="$(docker compose "${files[@]}" ps -q mongo)"
  wait_healthy "$cid"
  start_service DATABASE_TYPE=mongo MONGO_HOST=localhost MONGO_PORT=27017 MONGO_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  local n
  n="$(docker exec "$cid" mongosh agent_service --quiet --eval \
    "db.checkpoints.countDocuments({thread_id:'$SMOKE_THREAD_ID'})" 2>/dev/null | tr -d '[:space:]')" || true
  assert_positive_count "$n" "mongo checkpoint documents for this run's thread"
  stop_service
  docker compose "${files[@]}" down -v
}

smoke_agui() {
  echo "=== AG-UI endpoint ==="
  # AG-UI 与存储后端无关，因此使用默认 SQLite 检查点保存器即可，
  # 无需数据库容器。
  start_service
  local out
  out="$(cd scripts/agui-client && npm install --silent && \
    AGENT_URL=http://localhost:8080 node client.mjs "Tell me a joke!" chatbot)" || true
  echo "$out"
  # 仅正常退出还不够：确认流已完整结束并返回
  # 模拟模型的响应，而不是空结果或部分结果。
  if ! grep -q "RUN_FINISHED" <<<"$out"; then
    echo "  ✗ FAIL: AG-UI stream did not reach RUN_FINISHED"
    return 1
  fi
  if ! grep -q "This is a test response from the fake model." <<<"$out"; then
    echo "  ✗ FAIL: AG-UI did not return the expected assistant response"
    return 1
  fi
  echo "  ✓ verified: AG-UI streamed a complete run with the expected response"
  stop_service
}

smoke_langfuse() {
  echo "=== LangFuse tracing (self-hosted) ==="
  local pk="pk-lf-smoke-public" sk="sk-lf-smoke-secret"

  # 获取指定版本的 Langfuse 官方自托管 Compose 配置，不在仓库中维护副本。
  # 使用不带 --suffix 的 mktemp，以兼容 macOS/BSD；`docker compose -f`
  # 不要求特定文件扩展名。
  LANGFUSE_COMPOSE="$(mktemp)"
  echo "  fetching LangFuse compose @ $LANGFUSE_REF"
  if ! curl -sSL "https://raw.githubusercontent.com/langfuse/langfuse/$LANGFUSE_REF/docker-compose.yml" \
    -o "$LANGFUSE_COMPOSE"; then
    echo "  ✗ FAIL: could not fetch LangFuse compose"
    return 1
  fi

  # LANGFUSE_INIT_* 在首次启动时预置组织、项目、用户和已知 API 密钥，
  # 因此无需手动注册，下方密钥的值也是确定的。
  echo "  starting LangFuse stack (this pulls ~5GB the first time)..."
  LANGFUSE_INIT_ORG_ID=smoke-org LANGFUSE_INIT_ORG_NAME=smoke \
  LANGFUSE_INIT_PROJECT_ID=smoke-project LANGFUSE_INIT_PROJECT_NAME=smoke \
  LANGFUSE_INIT_PROJECT_PUBLIC_KEY="$pk" LANGFUSE_INIT_PROJECT_SECRET_KEY="$sk" \
  LANGFUSE_INIT_USER_EMAIL=smoke@example.com LANGFUSE_INIT_USER_NAME=smoke \
  LANGFUSE_INIT_USER_PASSWORD=smokepassword123 \
    docker compose -f "$LANGFUSE_COMPOSE" up -d

  echo "  waiting for langfuse-web..."
  for _ in $(seq 1 40); do
    curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1 && break
    sleep 3
  done
  if ! curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1; then
    echo "  ✗ FAIL: langfuse-web did not become ready"
    return 1
  fi

  start_service LANGFUSE_TRACING=true LANGFUSE_HOST=http://localhost:3000 \
    LANGFUSE_PUBLIC_KEY="$pk" LANGFUSE_SECRET_KEY="$sk"

  # （1）服务级检查：/health 对实例执行 langfuse.auth_check()。
  if curl -s http://localhost:8080/health | grep -q '"langfuse":"connected"'; then
    echo "  ✓ /health reports langfuse connected"
  else
    echo "  ✗ FAIL: /health did not report langfuse connected"
    return 1
  fi

  # （2）关键检查：启用追踪的调用必须在 Langfuse 中生成真实追踪记录。
  uv run python -c "
import sys; sys.path.insert(0, 'src')
from client import AgentClient
c = AgentClient('http://localhost:8080')
r = c.invoke('Trace me please', thread_id='$SMOKE_THREAD_ID', model='fake')
assert r.type == 'ai', r
print('  traced invoke ok')
"
  echo "  waiting for the trace to land in LangFuse (ingestion is async)..."
  local n=0
  for _ in $(seq 1 20); do
    n="$(curl -s -u "$pk:$sk" "http://localhost:3000/api/public/traces?limit=5" \
      | python3 -c 'import sys, json; print(len(json.load(sys.stdin).get("data", [])))' 2>/dev/null || echo 0)"
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )) && break
    sleep 3
  done
  assert_positive_count "$n" "LangFuse traces recorded for this run"

  stop_service
  docker compose -f "$LANGFUSE_COMPOSE" down -v
  rm -f "$LANGFUSE_COMPOSE"
  LANGFUSE_COMPOSE=""
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(postgres mongo agui)

# 将 "all" 展开为全部目标，包括较重的 langfuse。
expanded=()
for t in "${targets[@]}"; do
  if [[ "$t" == "all" ]]; then
    expanded+=(postgres mongo agui langfuse)
  else
    expanded+=("$t")
  fi
done
targets=("${expanded[@]}")

for t in "${targets[@]}"; do
  case "$t" in
    postgres) smoke_postgres ;;
    mongo)    smoke_mongo ;;
    agui)     smoke_agui ;;
    langfuse) smoke_langfuse ;;
    *) echo "unknown target: $t (valid: postgres, mongo, agui, langfuse, all)"; exit 2 ;;
  esac
done

echo "--- All smoke tests passed ---"
