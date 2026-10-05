"""对所有已配置的 LLM 服务商执行轻量级在线冒烟测试。

对于环境中已配置凭证的服务商，向 schema.models 中的每个模型发送
只生成一个 token 的最小请求，并报告各模型是否通过。
这是维护者定期刷新模型列表的工具（见 model-refresh 技能），不属于 pytest
测试套件，因为它会真实访问服务商 API，并产生少量费用。

用法（在仓库根目录运行；与 src/run_service.py 一样需要 PYTHONPATH=src）：
    PYTHONPATH=src uv run python scripts/check_live_models.py
    PYTHONPATH=src uv run python scripts/check_live_models.py --provider anthropic google

如果环境不允许设置 ANTHROPIC_API_KEY，可以把密钥放入其他变量，
再通过 --anthropic-api-key-env 指定：
    PYTHONPATH=src uv run python scripts/check_live_models.py --anthropic-api-key-env MY_VAR
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Callable


def _remap_anthropic_api_key(argv: list[str]) -> None:
    """如果指定了 --anthropic-api-key-env，将其对应变量复制到 ANTHROPIC_API_KEY。

    此操作先于 core.settings 的导入，因为 Settings 在构造时读取环境变量，
    无法等到下方主 argparse 解析过程再执行。
    仅在显式传入该参数时生效；否则保持 ANTHROPIC_API_KEY 的原有状态。
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_var = None
    for i, arg in enumerate(argv):
        if arg == "--anthropic-api-key-env" and i + 1 < len(argv):
            env_var = argv[i + 1]
            break
        if arg.startswith("--anthropic-api-key-env="):
            env_var = arg.split("=", 1)[1]
            break
    if env_var and os.environ.get(env_var):
        os.environ["ANTHROPIC_API_KEY"] = os.environ[env_var]


_remap_anthropic_api_key(sys.argv[1:])

from core.llm import get_model  # noqa: E402
from core.settings import settings  # noqa: E402
from schema.models import (  # noqa: E402
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
    VertexAIModelName,
)

PROMPT = "Reply with exactly one word: OK"

# 仅需 API 密钥或开关即可进行冒烟测试的服务商。不包含 Ollama、
# OpenAI 兼容模型和模拟模型：它们需要本地基础设施或专用配置，
# 无法只通过检查密钥是否设置来测试，因此不适合此通用扫描。
PROVIDER_MODELS: dict[Provider, tuple[type[AllModelEnum], Callable[[], bool]]] = {
    Provider.OPENAI: (OpenAIModelName, lambda: bool(settings.OPENAI_API_KEY)),
    Provider.ANTHROPIC: (AnthropicModelName, lambda: bool(settings.ANTHROPIC_API_KEY)),
    Provider.GOOGLE: (GoogleModelName, lambda: bool(settings.GOOGLE_API_KEY)),
    Provider.GROQ: (GroqModelName, lambda: bool(settings.GROQ_API_KEY)),
    Provider.DEEPSEEK: (DeepseekModelName, lambda: bool(settings.DEEPSEEK_API_KEY)),
    Provider.OPENROUTER: (OpenRouterModelName, lambda: bool(settings.OPENROUTER_API_KEY)),
    Provider.AWS: (AWSModelName, lambda: settings.USE_AWS_BEDROCK),
    Provider.AZURE_OPENAI: (AzureOpenAIModelName, lambda: bool(settings.AZURE_OPENAI_API_KEY)),
    Provider.VERTEXAI: (VertexAIModelName, lambda: bool(settings.GOOGLE_APPLICATION_CREDENTIALS)),
}


async def check_model(model_name: AllModelEnum) -> tuple[bool, str]:
    try:
        model = get_model(model_name)
        result = await model.ainvoke(PROMPT)
        return True, str(result.content).strip()[:60]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:120]


async def main(provider_filter: set[str]) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for provider, (model_enum, has_credentials) in PROVIDER_MODELS.items():
        if provider_filter and provider.value not in provider_filter:
            continue
        if not has_credentials():
            rows.append((provider.value, "-", "SKIP", "no credentials configured"))
            continue
        for model_name in model_enum:
            ok, detail = await check_model(model_name)
            rows.append((provider.value, model_name.value, "PASS" if ok else "FAIL", detail))

    name_width = max((len(r[1]) for r in rows), default=10)
    for provider, model, status, detail in rows:
        print(f"{status:5} {provider:12} {model:<{name_width}} {detail}")

    if any(r[2] == "FAIL" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        nargs="*",
        default=[],
        metavar="PROVIDER",
        help="Limit to these provider values (e.g. anthropic google). Default: all configured.",
    )
    parser.add_argument(
        "--anthropic-api-key-env",
        default=None,
        metavar="VAR",
        help=(
            "Env var to read the Anthropic key from if ANTHROPIC_API_KEY itself isn't set. "
            "No fallback is checked unless this is passed. Applied before this script's own "
            "imports run, so the value here is informational -- set the env var itself before "
            "invoking the script."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(set(args.provider)))
