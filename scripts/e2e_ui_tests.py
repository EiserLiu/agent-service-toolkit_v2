"""Streamlit 应用界面的浏览器端到端测试场景。

在 scripts/smoke_live_app.py 的单轮聊天测试基础上，覆盖曾因 Streamlit
升级或客户端、数据结构变更而出现回归的关键用户流程。通过真实浏览器模拟
用户操作，发现模拟传输层的 pytest 测试和 Docker CI 健康检查无法发现的问题。

用法：
    uv run --with playwright python scripts/e2e_ui_tests.py [URL] [scenario ...]

默认运行全部场景，也可指定场景：
    uv run --with playwright python scripts/e2e_ui_tests.py            # 在本地运行全部场景
    uv run --with playwright python scripts/e2e_ui_tests.py chat feedback
    uv run --with playwright python scripts/e2e_ui_tests.py https://my-app.example.com
    uv run --with playwright python scripts/e2e_ui_tests.py --list

应用地址由参数指定，默认为 http://localhost:8501。传入其他地址或设置
LIVE_APP_URL 即可测试已部署的应用。提交 PR 前可使用本地 USE_FAKE_MODEL=true
服务和 streamlit run，线上检查则使用部署地址。

默认场景使用 fake 模型，无需 API 密钥，也不会调用真实 LLM。要验证真实模型，
使用 --model=<name> 或 E2E_LIVE_MODEL 配合可选的 live_model 场景；
它会在设置中选择该模型并发送一条简短提示词，只产生一次低成本调用：
    uv run --with playwright python scripts/e2e_ui_tests.py --model=gpt-5-nano live_model

注意：
  - 各场景仅发送简短提示词，因此真实模型测试的费用也较低。
  - 反馈场景只验证控件渲染和可交互性，不提交评分；点击星级会经后端写入
    LangSmith，在每次监控运行时这样做会污染生产项目。其余场景也不会主动
    写入 LangSmith；普通聊天和历史查询仅在显式启用追踪时才产生追踪记录。

需要 Playwright 能找到 Chromium：执行 playwright install chromium，或通过
PLAYWRIGHT_BROWSERS_PATH 指向预装浏览器（如 Claude Code 云环境）。
全部所选场景通过时退出码为 0，否则为 1；失败时在当前工作目录保存
e2e_<scenario>_failure.png 以便排查。
"""

import os
import sys
import time
import urllib.parse

from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

DEFAULT_URL = "http://localhost:8501"
# Claude Code 云环境中预装浏览器的固定符号链接，
# 在当前 Playwright 自带的浏览器版本不可用时作为备用。
CLOUD_CHROMIUM = "/opt/pw-browsers/chromium"

# 使用简单且不调用工具的 Agent，使每轮对话恰好生成
# 一条助手消息，从而保证基于消息数量的等待不受模型影响。
# 专门测试 Agent 选择的场景会覆盖此设置。
CHAT_AGENT = "chatbot"

# 恢复会话的场景要求完整的多轮历史能从检查点还原。
# 使用 @entrypoint 的 chatbot 通过 /history 仅暴露最后一条回复，
# 因此这里使用消息通道会累积每轮内容的 StateGraph Agent。
HISTORY_AGENT = "research-assistant"

WAKE_TIMEOUT_S = 180
RESPONSE_TIMEOUT_S = 120
STREAM_SETTLE_S = 4

CHAT_INPUT = '[data-testid="stChatInput"] textarea'
CHAT_MESSAGE = '[data-testid="stChatMessage"]'


class E2EError(Exception):
    """场景断言失败。"""


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 浏览器和应用辅助函数
# --------------------------------------------------------------------------- #
def launch_browser(p) -> Browser:
    try:
        return p.chromium.launch()
    except Exception:
        executable = os.environ.get("CHROMIUM_EXECUTABLE", CLOUD_CHROMIUM)
        if not os.path.exists(executable):
            raise
        log(f"bundled browser missing - falling back to {executable}")
        return p.chromium.launch(executable_path=executable)


def build_url(base_url: str, **params: str) -> str:
    """将查询参数合并到 base_url，保留已有参数。"""
    parts = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    query.update({k: v for k, v in params.items() if v is not None})
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def wake_if_sleeping(page: Page) -> None:
    """Streamlit Community Cloud 会为休眠应用显示唤醒页面。"""
    wake_button = page.get_by_text("get this app back up", exact=False)
    try:
        wake_button.first.wait_for(state="visible", timeout=5_000)
    except PlaywrightTimeoutError:
        return  # 应用未休眠
    log("app is asleep - clicking wake-up button")
    wake_button.first.click()


def open_app(browser: Browser, url: str, agent: str | None = None) -> Page:
    """为应用打开新的浏览器上下文，并等待其可交互。"""
    if agent:
        url = build_url(url, agent=agent)
    page = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    wake_if_sleeping(page)
    # 聊天输入框出现，表示 Streamlit 已启动、WebSocket 已连接，
    # 应用脚本已执行（输入框在 Agent 和模型初始化后渲染）。
    page.locator(CHAT_INPUT).wait_for(state="visible", timeout=WAKE_TIMEOUT_S * 1_000)
    return page


def query_param(page: Page, key: str) -> str | None:
    query = urllib.parse.urlsplit(page.url).query
    return dict(urllib.parse.parse_qsl(query)).get(key)


def message_texts(page: Page) -> list[str]:
    return [(t or "").strip() for t in page.locator(CHAT_MESSAGE).all_inner_texts()]


def send_message(page: Page, text: str) -> None:
    chat_input = page.locator(CHAT_INPUT)
    chat_input.fill(text)
    chat_input.press("Enter")


def wait_for_response(
    page: Page,
    prompt: str,
    min_count: int,
    timeout_s: int = RESPONSE_TIMEOUT_S,
) -> str:
    """等待针对 prompt 的新助手回复出现并结束流式输出。

    模拟模型每轮回复相同，因此按消息数量而非文本内容识别新一轮。
    最后一条消息非空、不等于提示词本身，且在 STREAM_SETTLE_S 内
    不再变化时，视为回复完成。
    """
    deadline = time.monotonic() + timeout_s
    last_text, stable_since = "", None
    while time.monotonic() < deadline:
        texts = message_texts(page)
        # 只有确认提示词已成为前一条消息，且其后出现助手回复，
        # 才能开始将最后一条消息视为待检查的回复。
        if len(texts) >= min_count and any(prompt in t for t in texts[:-1]):
            text = texts[-1]
            if text and text != prompt:
                if text == last_text:
                    if stable_since and time.monotonic() - stable_since >= STREAM_SETTLE_S:
                        return text
                else:
                    last_text, stable_since = text, time.monotonic()
        time.sleep(1)
    raise E2EError(f"no stable assistant reply to {prompt!r} (>= {min_count} msgs) in {timeout_s}s")


def open_settings(page: Page) -> None:
    """打开设置弹出面板。面板在重新运行后仍保持打开，因此一次打开后完成
    所有设置操作再关闭；再次点击会将其关闭。
    """
    page.get_by_role("button", name="Settings").first.click()
    page.locator('[data-testid="stSelectbox"]').first.wait_for(state="visible", timeout=15_000)


def selectbox_value(page: Page, label: str) -> str | None:
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text=label)
    return box.get_by_role("combobox").get_attribute("value")


# --------------------------------------------------------------------------- #
# 测试场景
# --------------------------------------------------------------------------- #
def scenario_chat(browser: Browser, base_url: str) -> None:
    """基准场景：发送一条消息，确认助手流式返回回复并完成输出。"""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    log(f"assistant replied ({len(reply)} chars): {reply[:80]!r}")


def scenario_multi_turn_resume(browser: Browser, base_url: str) -> None:
    """进行两轮对话，再在新会话中通过分享链接恢复。

    验证会话持久化、恢复时按 Agent 查询 /history，以及分享/恢复对话框。
    #330 修复的对话框回归会在此失败，因为构建分享链接时会报错，无法渲染链接。
    """
    turn1 = "First turn: remember the number 7"
    turn2 = "Second turn: what number did I mention?"
    page = open_app(browser, base_url, agent=HISTORY_AGENT)
    thread_id = query_param(page, "thread_id")
    if not thread_id:
        raise E2EError("thread_id was not published to the URL on load")

    send_message(page, turn1)
    wait_for_response(page, turn1, min_count=2)
    send_message(page, turn2)
    wait_for_response(page, turn2, min_count=4)

    # 打开分享/恢复对话框，读取生成的分享链接。
    # 对话框框架会先于 Streamlit 流式传入的 Markdown 出现，因此需等待
    # 代码块中的文本，而不能在对话框刚打开时立即读取。
    page.get_by_role("button", name="Share/resume chat").first.click()
    dialog = page.locator('[role="dialog"]')
    dialog.wait_for(state="visible", timeout=15_000)
    code = dialog.locator('[data-testid="stCode"] code')
    share_url = ""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if code.count() and (share_url := code.first.inner_text().strip()):
            break
        page.wait_for_timeout(500)
    if not share_url:
        raise E2EError("Share/resume dialog rendered no chat URL (share_chat_dialog broken?)")
    if thread_id not in share_url or "agent=" not in share_url:
        raise E2EError(f"share URL missing thread_id/agent: {share_url!r}")
    log(f"share URL: {share_url}")

    # 在没有共享状态的新会话中恢复对话，并确认历史已还原：
    # StateGraph Agent 会持久化每轮消息，因此
    # 之前的两条提示词都应该重新显示。（全新或空的会话
    # 只会显示 Agent 的欢迎消息。）
    resumed = open_app(browser, share_url)
    if query_param(resumed, "thread_id") != thread_id:
        raise E2EError("resumed session did not carry the original thread_id from the share URL")
    joined = "\n".join(message_texts(resumed))
    for needle in (turn1, turn2):
        if needle not in joined:
            raise E2EError(f"resumed thread did not replay {needle!r} - history not restored")
    log(f"resumed thread {thread_id} replayed both prior turns from history")


def scenario_settings_selectors(browser: Browser, base_url: str) -> None:
    """验证设置面板中的模型和 Agent 选择框能渲染，且切换非默认 Agent
    后会同步更新 URL 中的 ?agent= 参数。
    """
    page = open_app(browser, base_url)  # 使用默认 Agent，因此初始 URL 没有 ?agent= 参数
    open_settings(page)

    model = selectbox_value(page, "LLM to use")
    if not model:
        raise E2EError("LLM selectbox rendered without a selected model")
    log(f"model selectbox shows: {model!r}")

    default_agent = selectbox_value(page, "Agent to use")
    agents = page.locator('[data-testid="stSelectbox"]').filter(has_text="Agent to use")
    agents.get_by_role("combobox").click()
    options = page.locator('[role="option"]')
    options.first.wait_for(state="visible", timeout=10_000)
    all_agents = [options.nth(i).inner_text().strip() for i in range(options.count())]
    if len(all_agents) < 2:
        raise E2EError(f"expected multiple agents to choose from, saw {all_agents}")
    # 选择任意非默认 Agent；默认 Agent 不会写入 URL，
    # 所以切换到非默认 Agent 才能验证查询参数绑定是否生效。
    target = next(a for a in all_agents if a != default_agent)
    options.filter(has_text=target).first.click()
    page.wait_for_timeout(1_500)

    if selectbox_value(page, "Agent to use") != target:
        raise E2EError(f"agent selectbox did not switch to {target!r}")
    if query_param(page, "agent") != target:
        raise E2EError(f"?agent= URL param is {query_param(page, 'agent')!r}, expected {target!r}")
    log(f"agent switched {default_agent!r} -> {target!r} and synced to the URL")


def scenario_feedback(browser: Browser, base_url: str) -> None:
    """回复后，星级反馈控件应正常渲染并可交互。

    仅断言容易受 Streamlit 升级影响的控件结构，不提交评分：点击星级
    会经后端写入 LangSmith，持续监控时会污染生产项目，且在后端无法访问
    LangSmith 时会卡住。验证星级已渲染、具备预期 aria-label，且已启用、可点击。
    """
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    wait_for_response(page, prompt, min_count=2)

    widget = page.locator('[data-testid="stFeedback"]').last
    widget.wait_for(state="visible", timeout=15_000)
    stars = widget.locator('[data-testid="stFeedbackButton"]')
    if stars.count() != 5:
        raise E2EError(f"expected a 5-star feedback widget, found {stars.count()} stars")
    labels = [stars.nth(i).get_attribute("aria-label") for i in range(5)]
    if labels != [f"{i} out of 5 stars" for i in range(1, 6)]:
        raise E2EError(f"feedback stars have unexpected aria-labels: {labels}")
    if not stars.first.is_enabled() or not stars.last.is_enabled():
        raise E2EError("feedback stars rendered but are not interactive")
    log("5-star feedback widget rendered with expected labels and is interactive")


def scenario_streaming_toggle(browser: Browser, base_url: str) -> None:
    """关闭“Stream results”后，仍可通过非流式 ainvoke 路径获得回复；
    默认流式测试不会覆盖此路径。
    """
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    open_settings(page)
    toggle = page.locator('[data-testid="stCheckbox"]').filter(has_text="Stream results")
    toggle.wait_for(state="visible", timeout=10_000)
    toggle.click()  # 默认开启，将其关闭
    page.keyboard.press("Escape")  # 关闭弹出面板，以便访问聊天输入框
    page.wait_for_timeout(500)

    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    log(f"non-streaming reply rendered ({len(reply)} chars)")


def scenario_new_chat(browser: Browser, base_url: str) -> None:
    """“New Chat”开启全新会话：URL 中生成新的 thread_id，并清空对话。"""
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    prompt = "Reply with the single word: pong"
    send_message(page, prompt)
    wait_for_response(page, prompt, min_count=2)
    old_thread = query_param(page, "thread_id")

    page.get_by_role("button", name="New Chat").first.click()
    page.locator(CHAT_INPUT).wait_for(state="visible", timeout=30_000)

    deadline = time.monotonic() + 15
    while query_param(page, "thread_id") == old_thread and time.monotonic() < deadline:
        page.wait_for_timeout(500)
    new_thread = query_param(page, "thread_id")
    if not new_thread or new_thread == old_thread:
        raise E2EError(f"thread_id did not change on New Chat (still {old_thread!r})")
    if any(prompt in t for t in message_texts(page)):
        raise E2EError("previous conversation was not cleared after New Chat")
    log(f"New Chat reset thread {old_thread} -> {new_thread} and cleared the conversation")


def scenario_live_model(browser: Browser, base_url: str) -> None:
    """可选场景：在设置中选择真实模型，确认端到端回复正常。

    其他场景使用 fake 模型；本场景调用真实 LLM，需要后端提供该模型及
    对应凭证。通过 --model=<name> 或 E2E_LIVE_MODEL 指定模型，发送一条
    简短提示词，只产生一次低成本调用，并确认回复不是模拟模型的占位文本。
    """
    model = os.environ.get("E2E_LIVE_MODEL", "").strip()
    if not model:
        raise E2EError("live_model needs a model - pass --model=<name> or set E2E_LIVE_MODEL")
    page = open_app(browser, base_url, agent=CHAT_AGENT)
    open_settings(page)
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text="LLM to use")
    box.get_by_role("combobox").click()
    option = page.get_by_role("option", name=model, exact=True)
    if not option.count():
        available = page.locator('[role="option"]').all_inner_texts()
        raise E2EError(f"model {model!r} is not offered by this app; available: {available}")
    option.first.click()
    page.keyboard.press("Escape")  # 关闭弹出面板，以便访问聊天输入框
    page.wait_for_timeout(500)

    prompt = "Reply with only the word: pong"
    send_message(page, prompt)
    reply = wait_for_response(page, prompt, min_count=2)
    if "fake model" in reply.lower():
        raise E2EError(f"expected a reply from {model!r} but got the fake-model placeholder")
    log(f"live model {model!r} replied ({len(reply)} chars): {reply[:80]!r}")


# 默认测试套件使用模拟模型，无需 API 密钥；live_model
# 会调用真实 LLM，仅在显式选择该场景或指定 --model 时运行。
SCENARIOS = {
    "chat": scenario_chat,
    "multi_turn_resume": scenario_multi_turn_resume,
    "settings_selectors": scenario_settings_selectors,
    "feedback": scenario_feedback,
    "streaming_toggle": scenario_streaming_toggle,
    "new_chat": scenario_new_chat,
    "live_model": scenario_live_model,
}
DEFAULT_SCENARIOS = [name for name in SCENARIOS if name != "live_model"]


# --------------------------------------------------------------------------- #
# 测试运行器
# --------------------------------------------------------------------------- #
def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        print("\n".join(SCENARIOS))
        return

    url = os.environ.get("LIVE_APP_URL", DEFAULT_URL)
    names = []
    for arg in args:
        if arg in SCENARIOS:
            names.append(arg)
        elif arg.startswith("--model="):
            os.environ["E2E_LIVE_MODEL"] = arg.split("=", 1)[1]
        elif arg.startswith(("http://", "https://")):
            url = arg
        else:
            print(f"unknown argument: {arg!r} (scenarios: {', '.join(SCENARIOS)})")
            sys.exit(2)
    if not names:
        # 默认运行模拟模型测试套件；仅在指定模型时额外运行 live_model。
        names = list(DEFAULT_SCENARIOS)
        if os.environ.get("E2E_LIVE_MODEL"):
            names.append("live_model")

    log(f"target: {url}")
    log(f"scenarios: {', '.join(names)}")

    results: dict[str, str] = {}
    with sync_playwright() as p:
        browser = launch_browser(p)
        for name in names:
            log(f"--- {name} ---")
            start = time.monotonic()
            try:
                SCENARIOS[name](browser, url)
                results[name] = "PASS"
                log(f"PASS: {name} ({time.monotonic() - start:.0f}s)")
            except Exception as e:
                results[name] = f"FAIL: {e}"
                log(f"FAIL: {name}: {e}")
                _screenshot_failure(browser, name)
        browser.close()

    log("=" * 60)
    for name in names:
        log(f"{results[name].split(':')[0]:<4} {name}: {results[name]}")
    failed = [n for n, r in results.items() if not r.startswith("PASS")]
    if failed:
        log(f"{len(failed)} of {len(names)} scenario(s) failed: {', '.join(failed)}")
        sys.exit(1)
    log(f"all {len(names)} scenario(s) passed")


def _screenshot_failure(browser: Browser, name: str) -> None:
    """场景失败时，为最后打开的页面保存截图。"""
    path = f"e2e_{name}_failure.png"
    try:
        contexts = browser.contexts
        if contexts and contexts[-1].pages:
            contexts[-1].pages[-1].screenshot(path=path, full_page=True)
            log(f"screenshot saved to {path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
