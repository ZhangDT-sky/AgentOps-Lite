from src.rag.vector_store import VectorStoreFactory


class Retriever:
    def __init__(self):
        self._vectorstore = None
        self._retriever = None

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = VectorStoreFactory.create()
        return self._vectorstore

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever()
        return self._retriever

    def retrieve(self, query: str):
        return self.retriever.get_relevant_documents(query)


retriever = Retriever()
