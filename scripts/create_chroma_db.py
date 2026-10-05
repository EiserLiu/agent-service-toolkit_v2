import os
import shutil

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# 从 .env 文件加载环境变量
load_dotenv()


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    # 初始化 Chroma 向量存储
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 根据文件扩展名加载文档
        # 按需添加其他加载器，例如 JSONLoader、TxtLoader 等
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # 跳过不支持的文件类型

        # 加载文档并分割为文本块
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # 将文本块加入 Chroma 向量存储
        for chunk in chunks:
            chunk_id = chroma.add_documents([chunk])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


if __name__ == "__main__":
    # 存放文档的文件夹路径
    folder_path = "./data"

    # 创建 Chroma 数据库
    chroma = create_chroma_db(folder_path=folder_path)

    # 从 Chroma 数据库创建检索器
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # 执行相似度搜索
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # 显示结果
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
