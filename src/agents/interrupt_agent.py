import logging
from datetime import datetime
from typing import Any

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

# 添加日志记录器
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` 来自 PEP 589 规范。

    文档：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[AgentState, Any]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one sentence summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """此节点演示在中断之前执行工作。"""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    birthdate: str | None = Field(
        description="The extracted birthdate in YYYY-MM-DD format. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


async def determine_birthdate(
    state: AgentState, config: RunnableConfig, store: BaseStore
) -> AgentState:
    """此节点检查对话历史以确定用户的出生日期，并优先查询存储。"""

    # 尝试获取 user_id，以便按用户隔离存储
    user_id = config["configurable"].get("user_id")
    logger.info(f"[determine_birthdate] Extracted user_id: {user_id}")
    namespace = None
    key = "birthdate"
    birthdate = None  # 初始化出生日期

    if user_id:
        # 在命名空间中使用 user_id，确保不同用户的数据互相隔离
        namespace = (user_id,)

        # 检查存储中是否已有该用户的出生日期
        try:
            result = await store.aget(namespace, key=key)
            # 兼容 store.aget 直接返回 Item 或返回列表的情况
            user_data = None
            if result:  # 检查是否有返回结果
                if isinstance(result, list):
                    if result:  # 检查列表是否非空
                        user_data = result[0]
                else:  # 否则将结果视为直接返回的 Item 对象
                    user_data = result

            if user_data and user_data.value.get("birthdate"):
                # 将 ISO 格式字符串转换回 datetime 对象
                birthdate_str = user_data.value["birthdate"]
                birthdate = datetime.fromisoformat(birthdate_str) if birthdate_str else None
                # 已获取出生日期，直接返回
                logger.info(
                    f"[determine_birthdate] Found birthdate in store for user {user_id}: {birthdate}"
                )
                return {
                    "birthdate": birthdate,
                    "messages": [],
                }
        except Exception as e:
            # 记录错误，处理存储可能不可用的情况
            logger.error(f"Error reading from store for namespace {namespace}, key {key}: {e}")
            # 读取失败时继续尝试提取
            pass
    else:
        # 没有 user_id 时，无法可靠地保存或读取特定用户的数据。
        # 可考虑记录这种情况。
        logger.warning(
            "Warning: user_id not found in config. Skipping persistent birthdate storage/retrieval for this run."
        )

    # 未从存储中获取出生日期时，继续尝试提取
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    ).with_config(tags=["skip_stream"])
    response: BirthdateExtraction = await model_runnable.ainvoke(state, config)

    # 提取后仍未找到出生日期，则中断执行
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\nPlease tell me your birthdate?")
        # 使用新输入重新提取
        state["messages"].append(HumanMessage(birthdate_input))
        # 注意：递归调用需要谨慎处理深度和状态更新
        return await determine_birthdate(state, config, store)

    # 已找到出生日期，将字符串转换为 datetime
    try:
        birthdate = datetime.fromisoformat(response.birthdate)
    except ValueError:
        # 解析失败时，请用户补充说明
        birthdate_input = interrupt(
            "I couldn't understand the date format. Please provide your birthdate in YYYY-MM-DD format."
        )
        # 使用新输入重新提取
        state["messages"].append(HumanMessage(birthdate_input))
        # 注意：递归调用需要谨慎处理深度和状态更新
        return await determine_birthdate(state, config, store)

    # 仅在存在 user_id 时保存新提取的出生日期
    if user_id and namespace:
        # 将 datetime 转为 ISO 格式字符串，以便 JSON 序列化
        birthdate_str = birthdate.isoformat() if birthdate else None
        try:
            await store.aput(namespace, key, {"birthdate": birthdate_str})
        except Exception as e:
            # 记录错误，处理存储写入可能失败的情况
            logger.error(f"Error writing to store for namespace {namespace}, key {key}: {e}")

    # 返回确定的出生日期（来自存储或本次提取）
    logger.info(f"[determine_birthdate] Returning birthdate {birthdate} for user {user_id}")
    return {
        "birthdate": birthdate,
        "messages": [],
    }


response_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant.

Known information:
- The user's birthdate is {birthdate_str}

User's latest message: "{last_user_message}"

Based on the known information and the user's message, provide a helpful and relevant response.
If the user asked for their birthdate, confirm it.
If the user asked for their zodiac sign, calculate it and tell them.
Otherwise, respond conversationally based on their message.
""")


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """根据用户的问题和已知出生日期生成最终回复。"""
    birthdate = state.get("birthdate")
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        last_user_message = state["messages"][-1].content
    else:
        last_user_message = ""

    if not birthdate:
        # 如果 determine_birthdate 正常工作并在必要时中断，通常不会进入此分支。
        # 这里处理出生日期仍然缺失的情况。
        return {
            "messages": [
                AIMessage(
                    content="I couldn't determine your birthdate. Could you please provide it?"
                )
            ]
        }

    birthdate_str = birthdate.strftime("%B %d, %Y")  # 格式化为展示文本

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, response_prompt.format(birthdate_str=birthdate_str, last_user_message=last_user_message)
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# 定义图
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("generate_response", generate_response)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "generate_response")
agent.add_edge("generate_response", END)

interrupt_agent = agent.compile()
interrupt_agent.name = "interrupt-agent"
