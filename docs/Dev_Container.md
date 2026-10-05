# 使用 VS Code 开发容器

这套环境包含一个 Python 开发容器和一个独立的 PostgreSQL 容器。
开发容器安装运行依赖、pytest、Ruff、Pyrefly、Git 和 SSH 客户端；应用由你在终端手动启动，方便查看日志和调试。

## 打开项目

1. 启动 Docker Desktop，并开启 Ubuntu-24.04 的 WSL Integration。
2. 在 Windows 的 VS Code 安装 **WSL** 和 **Dev Containers** 扩展。
3. 保留根目录 `.env` 中已配置的 MiMo 参数；开发环境自动覆盖数据库连接参数，使用独立开发数据库。
4. 在 WSL 中执行：

   ```bash
   cd /mnt/c/Users/12560/Desktop/agent-service-toolkit
   code .
   ```

5. 在 VS Code 按 `F1`，选择 **Dev Containers: Reopen in Container**。
6. 等待镜像构建、依赖同步与扩展安装完成。左下角应显示开发容器名称。

容器工作目录是 `/workspaces/agent-service-toolkit`，Python 解释器是 `/opt/venv/bin/python`。
无需在 WSL 或 Windows 再安装项目 Python 依赖。

## 挂载与持久化

| 内容 | 容器路径 | 保存位置 |
| --- | --- | --- |
| 项目源码（包括 Git 和本地 `.env`） | `/workspaces/agent-service-toolkit` | 宿主机原项目目录，双向共享 |
| Python 虚拟环境 | `/opt/venv` | `dev_venv` 命名卷 |
| uv 下载缓存 | `/home/vscode/.cache/uv` | `uv_cache` 命名卷 |
| 开发数据库 | PostgreSQL 的 `/var/lib/postgresql/data` | `postgres_dev_data` 命名卷 |

在容器中编辑、添加或删除源码，也会修改宿主机项目。关闭容器后文件仍然保留。
虚拟环境独立于源码挂载，避免 Windows 和 Linux 共用 `.venv`。
开发数据库使用独立 Compose 项目，不包含之前普通运行环境的聊天记录。
保留数据时不要给 `docker compose down` 添加 `-v`。

## 启动应用

在 VS Code **容器终端**中运行后端：

```bash
python src/run_service.py
```

再开一个容器终端，运行界面：

```bash
streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

浏览器访问：

- 开发 API 文档：<http://localhost:8081/docs>
- 开发聊天界面：<http://localhost:8502>

容器内部仍使用 8080 和 8501；宿主机映射为 8081 和 8502，避免与原环境冲突。
开发容器空闲时只运行 `sleep infinity`，启动应用前访问上述端口不会有网页。
后端启用开发重载，Streamlit 使用轮询监测文件，适配当前 `/mnt/c` 挂载目录。

## 安装依赖与运行检查

修改依赖后，在容器终端运行：

```bash
uv sync --frozen --no-install-project
```

有意新增依赖时使用 `uv add 包名`（开发工具使用 `uv add --dev 包名`），并检查 `pyproject.toml` 与 `uv.lock` 的差异。

按改动范围运行检查，例如：

```bash
pytest tests/schema
ruff check src
pyrefly check
```

完整测试命令为 `pytest`。默认测试不等于真实模型或 Docker 集成验证。
学习时先选相关的无外部调用测试，不要把真实模型可用性与单元测试混为一谈。

## Git 与 SSH

VS Code Dev Containers 可以转发宿主环境的 SSH agent，无需把私钥复制进镜像。
如需从容器推送，在打开 VS Code 前先在 WSL 启动 agent 并加载已有密钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
code .
```

容器内可用 `ssh -T git@github.com` 检查认证。检查 `git config user.name` 和 `git config user.email`，确认提交署名正确。

## 更新环境与排错

- 修改 Dockerfile / Dev Container 配置后，使用 **Dev Containers: Rebuild Container**。
- 修改 `.env` 后也需要重建容器配置以更新环境变量；命名卷会保留。
- 虚拟环境卷会跨重建保留，`postCreateCommand` 会重新同步锁定依赖。
- 更换 Python 大版本时，已有环境可能不再兼容，应单独重建虚拟环境卷；不要删除数据库卷。
- 当前代码在 Windows 文件系统，首次依赖安装或文件扫描可能较慢；以后可迁到 WSL 的 `~/projects/`，容器配置无需改挂载路径。

在 **WSL 宿主终端**中可单独检查或启动开发环境：

```bash
docker compose -f .devcontainer/compose.yaml config --quiet
docker compose -f .devcontainer/compose.yaml up -d --build
docker compose -f .devcontainer/compose.yaml exec dev bash
```

如果 VS Code 采用了不同的 Compose 项目名，请以 Dev Containers 日志中的 `-p` 参数为准。
使用终端创建的容器时，也可以通过 **Dev Containers: Reopen in Container** 加载项目配置。

## 参考

- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container)
- [uv 容器开发与虚拟环境](https://docs.astral.sh/uv/guides/integration/docker/)
