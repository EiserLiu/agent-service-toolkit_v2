import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings


def calculator_func(expression: str) -> str:
    # 工具说明：使用 numexpr 计算数学表达式。仅用于数学问题，输入必须是合法的 numexpr 数学表达式。expression 为表达式字符串，返回值为计算结果字符串。
    # 下方英文文档会作为工具元数据传给模型，保留原文以维持调用行为。
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # 限制访问全局变量
                local_dict=local_dict,  # 添加常用数学函数
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# 格式化检索到的文档
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # 为项目说明数据库创建嵌入函数
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # 加载已保存的向量数据库
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    # 工具说明：在 chroma_db 中搜索公司手册中的信息。
    # 下方英文文档会作为工具元数据传给模型，保留原文以维持调用行为。
    """Searches chroma_db for information in the company's handbook."""
    # 获取 Chroma 检索器
    retriever = load_chroma_db()

    # 在数据库中搜索相关文档
    documents = retriever.invoke(query)

    # 将文档格式化为字符串
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # 根据数据库的用途修改名称
