from typing import List
from src.agent.state import State
from src.rag.retriever import retriever


class RetrievalNode:
    """
    RAG 检索节点：从向量数据库检索相关文档
    
    职责：
    - 根据 user_query 进行向量检索
    - 将检索结果写入 state.retrieved_docs
    - 处理检索失败的情况
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def run(self, state: State) -> dict:
        """
        执行向量检索
        """
        # 检查是否需要检索
        if state.need_retrieval is False:
            return {"retrieved_docs": []}
        
        try:
            # 使用 retriever 进行检索
            # retriever.retrieve() 返回 LangChain Document 对象列表
            documents = retriever.retrieve(state.user_query)
            
            # 将 Document 对象转换为字符串列表
            # Document 对象有 page_content 属性存储文档内容
            retrieved_texts = []
            for doc in documents[:self.top_k]:  # 限制返回数量
                # 处理 Document 对象
                if hasattr(doc, 'page_content'):
                    retrieved_texts.append(doc.page_content)
                elif isinstance(doc, str):
                    retrieved_texts.append(doc)
                else:
                    # 如果是不认识的类型，尝试转换为字符串
                    retrieved_texts.append(str(doc))
            
            # 更新状态
            state.retrieved_docs = retrieved_texts
            
            # 记录检索信息到 memory（用于调试和可观测性）
            state.memory["retrieval_count"] = len(retrieved_texts)
            state.memory["retrieval_success"] = True
            
            return {"retrieved_docs": retrieved_texts}
            
        except Exception as e:
            # 检索失败时的处理
            error_msg = str(e)
            state.memory["retrieval_error"] = error_msg
            state.memory["retrieval_success"] = False
            
            # 返回空列表，不阻塞流程
            # 后续节点可以根据 retrieved_docs 是否为空判断检索是否成功
            return {"retrieved_docs": []}
