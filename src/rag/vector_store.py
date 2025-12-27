from langchain_community.vectorstores import FAISS
from src.agent.models import embedding


class VectorStoreFactory:
    """
    向量存储初始化与加载
    """

    @staticmethod
    def create():
        # MVP 阶段直接使用内存 FAISS
        # 确保 texts 是纯字符串列表
        texts = [
            "AgentOps-Lite 是一个轻量级 Agent 可观测性系统",
            "RAG 包含检索、重写、召回与排序等阶段",
        ]
        # 确保所有元素都是字符串类型，且不为空
        texts = [str(text).strip() for text in texts if text and str(text).strip()]
        
        if not texts:
            raise ValueError("初始化向量库时，texts 列表不能为空")
        
        # 使用 from_texts 方法创建向量库
        # 注意：DashScope API 可能需要确保 texts 是纯字符串列表
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embedding,
        )
        return vectorstore
