import os
from dotenv import load_dotenv
load_dotenv('.env.local')

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain.tools import tool
from pydantic import BaseModel, Field

class DataRetriever:
    def __init__(self, data_path, embeddings_model=None, api_key=None, base_url=None):
        self.data_path = data_path
        self.embeddings_model = OpenAIEmbeddings(
            model= embeddings_model if embeddings_model else "BAAI/bge-large-zh-v1.5",
            api_key= api_key if api_key else os.getenv("SF_API_KEY") , 
            base_url= base_url if base_url else os.getenv("SF_BASE_URL"),
        )

    def load_directory(self, directory):
        loader = DirectoryLoader(directory)
        documents = loader.load()
        return documents
    
    def load_data(self):
        # 加载文档
        loader = UnstructuredWordDocumentLoader(self.data_path)
        pages = loader.load_and_split()
        # 文档切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        texts = text_splitter.create_documents(
            [page.page_content for page in pages]
        )
        return texts
    def get_retriever(self):
        texts = self.load_data()
        # 灌库
        db = FAISS.from_documents(texts, self.embeddings_model)
        # 检索 top-3 结果
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever

data_retriever = DataRetriever(r'.\data\杨帆中学比赛解说词2.0.docx').get_retriever()

class RagInput(BaseModel):
    query_description: str = Field(
        description="关于学校基础设施的查询描述",
        example="学生宿舍怎么样？"
    )

@tool('rag_tool', args_schema=RagInput)
def rag_query(query_description: str):
    """查询学校基础设施情况"""
    docs = data_retriever.invoke(query_description)
    rst = [f'\n{i}.' + doc.page_content for i, doc in enumerate(docs)]
    print(f'get rag data: {rst}')
    return rst

if __name__ == "__main__":
    retriever = DataRetriever(r'.\data\杨帆中学比赛解说词2.0.docx').get_retriever()

    docs = retriever.invoke('人工智能')

    for doc in docs:
        print(doc.page_content)
        print("----")