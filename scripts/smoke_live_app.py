"""通过真实浏览器对已部署的 Streamlit 应用执行端到端冒烟测试。

加载应用，发送一条聊天消息，确认助手以流式方式返回回复。Streamlit
依赖 WebSocket，普通 HTTP 检查只能证明页面外壳可加载；本测试覆盖完整
聊天链路（浏览器 → Streamlit → Agent 服务 → LLM → 原路返回）。

用法：
    uv run --with playwright python scripts/smoke_live_app.py [URL]

默认使用已部署应用的地址，也可设置 LIVE_APP_URL。本地测试示例：
    uv run --with playwright python scripts/smoke_live_app.py http://localhost:8501

需要 Playwright 能找到 Chromium：执行 playwright install chromium，或通过
PLAYWRIGHT_BROWSERS_PATH 指向预装浏览器（如 Claude Code 云环境）。
成功退出码为 0，失败为 1；失败时在当前工作目录保存
smoke_live_app_failure.png 以便排查。

注意：测试线上应用会发送一条真实消息，产生一次低成本 LLM 调用。
对于休眠的 Streamlit Community Cloud 应用，会点击唤醒页面，可能需等待几分钟。
"""

import os
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://agent-service-toolkit.streamlit.app/"
# Claude Code 云环境中预装浏览器的固定符号链接，
# 在当前 Playwright 自带的浏览器版本不可用时作为备用。
CLOUD_CHROMIUM = "/opt/pw-browsers/chromium"
TEST_MESSAGE = "Reply with the single word: pong"
CHAT_INPUT_SELECTOR = '[data-testid="stChatInput"] textarea'
WAKE_TIMEOUT_S = 180
RESPONSE_TIMEOUT_S = 120
STREAM_SETTLE_S = 4


def log(msg: str) -> None:
    print(f"[smoke_live_app] {msg}", flush=True)


def _dump_testids(ctx, label: str) -> None:
    try:
        ids = ctx.locator("[data-testid]").evaluate_all(
            "els => Array.from(new Set(els.map(e => e.getAttribute('data-testid')))).slice(0, 40)"
        )
        log(f"data-testids in {label} ({len(ids)}): {ids}")
    except Exception as e:
        log(f"data-testids in {label} unavailable: {e}")


def dump_diagnostics(page) -> None:
    """生成失败运行的文本诊断信息，便于直接从 CI 日志阅读。"""
    try:
        log(f"page title: {page.title()!r}  url: {page.url}")
    except Exception:
        pass
    n_iframes = page.locator("iframe").count()
    log(f"iframes present: {n_iframes}")
    _dump_testids(page, "top frame")
    for i in range(n_iframes):
        _dump_testids(page.frame_locator("iframe").nth(i), f"iframe[{i}]")
    try:
        body = (page.locator("body").inner_text(timeout=2_000) or "").strip()
        log(f"top-frame visible text ({len(body)} chars): {body[:600]!r}")
    except Exception:
        pass


def fail(page, reason: str) -> None:
    log(f"FAIL: {reason}")
    try:
        dump_diagnostics(page)
    except Exception as e:
        log(f"diagnostics unavailable: {e}")
    try:
        page.screenshot(path="smoke_live_app_failure.png", full_page=True)
        log("screenshot saved to smoke_live_app_failure.png")
    except Exception:
        pass
    sys.exit(1)


def wake_if_sleeping(page) -> None:
    """Streamlit Community Cloud 会为休眠应用显示唤醒页面。"""
    wake_button = page.get_by_text("get this app back up", exact=False)
    try:
        wake_button.first.wait_for(state="visible", timeout=5_000)
    except PlaywrightTimeoutError:
        return  # 应用未休眠
    log("app is asleep - clicking wake-up button")
    wake_button.first.click()


def find_app_root(page, timeout_s: int):
    """返回包含应用聊天输入框的页面或 iframe 上下文。

    本地应用渲染在顶层页面；Streamlit Community Cloud 将应用放入 iframe，
    page.locator() 无法直接访问其内部，因此也需探测 iframe。
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if page.locator(CHAT_INPUT_SELECTOR).count():
            return page
        for i in range(page.locator("iframe").count()):
            frame = page.frame_locator("iframe").nth(i)
            if frame.locator(CHAT_INPUT_SELECTOR).count():
                return frame
        time.sleep(1)
    return None


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LIVE_APP_URL", DEFAULT_URL)
    log(f"target: {url}")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            executable = os.environ.get("CHROMIUM_EXECUTABLE", CLOUD_CHROMIUM)
            if not os.path.exists(executable):
                raise
            log(f"bundled browser missing - falling back to {executable}")
            browser = p.chromium.launch(executable_path=executable)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        wake_if_sleeping(page)

        # 聊天输入框出现，表示 Streamlit 已启动、WebSocket 已连接，
        # 应用脚本已执行（输入框在 Agent 和模型初始化后渲染）。判断
        # 应用位于顶层页面（本地）还是 Community Cloud 的
        # iframe 中，并在对应上下文中执行所有应用交互。
        root = find_app_root(page, WAKE_TIMEOUT_S)
        if root is None:
            fail(page, f"chat input never appeared within {WAKE_TIMEOUT_S}s")
        log(f"app loaded, chat input visible ({'top-level' if root is page else 'iframe'})")

        chat_input = root.locator(CHAT_INPUT_SELECTOR)
        messages = root.locator('[data-testid="stChatMessage"]')
        # 欢迎消息仅在空会话中渲染，发送消息后重新运行时会消失，
        # 因此按文本而非消息数量进行检测。
        pre_send_last = (messages.last.inner_text() or "").strip() if messages.count() else ""

        chat_input.fill(TEST_MESSAGE)
        chat_input.press("Enter")
        log("message sent, waiting for assistant response")

        # 预期会话中先出现我们发送的消息，随后出现一条
        # 非空、全新且内容稳定的助手最终回复（流式输出已完成）。
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        last_text, stable_since = "", None
        while time.monotonic() < deadline:
            texts = [(t or "").strip() for t in messages.all_inner_texts()]
            if texts and any(TEST_MESSAGE in t for t in texts[:-1]):
                text = texts[-1]
                if text and text != TEST_MESSAGE and text != pre_send_last:
                    if text == last_text:
                        if stable_since and time.monotonic() - stable_since >= STREAM_SETTLE_S:
                            log(f"PASS: assistant responded ({len(text)} chars): {text[:120]!r}")
                            browser.close()
                            return
                    else:
                        last_text, stable_since = text, time.monotonic()
            time.sleep(1)

        fail(page, f"no stable assistant response within {RESPONSE_TIMEOUT_S}s")


if __name__ == "__main__":
    main()
