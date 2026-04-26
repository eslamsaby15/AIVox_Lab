"""Q&A controller with vector-store retrieval."""

from __future__ import annotations

from typing import Optional

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .BaseController import BaseController
from ..Stores.LLM import GenAIProvider


class QAController(BaseController):
    def __init__(
        self,
        provider: GenAIProvider,
        embedding_model,
        text: str,
    ) -> None:
        super().__init__()

        self.provider = provider
        self.embedding_model = embedding_model

        self.template = "\n".join([
            "You are an assistant for question-answering tasks.",
            "Use the following retrieved context to answer the question.",
            "If you don't know the answer, say so honestly.",
            "Keep your answer concise — 3 sentences maximum.\n",
            "Question: {question}\n",
            "Context: {context}",
        ])

        self.text = text
        self.vector_db = self._process_documents(text=self.text)

    # ── Document processing ───────────────────────────────────────────────────

    def _get_chunks(self, text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def _build_vectordb(self, chunks: list[str]):
        return InMemoryVectorStore.from_texts(chunks, self.embedding_model)

    def _process_documents(self, text: str):
        chunks = self._get_chunks(text=text)
        return self._build_vectordb(chunks=chunks)

    # Public aliases kept for backward compatibility
    def get_chunks(self, text: str, chunk_size: int = 400, chunk_overlap: int = 50):
        return self._get_chunks(text, chunk_size, chunk_overlap)

    def get_vectordb(self, chunks: list[str]):
        return self._build_vectordb(chunks)

    def process_documents(self, text: str):
        return self._process_documents(text)

    # ── QA ────────────────────────────────────────────────────────────────────

    def generate_answer(self, query: str, top_k: int = 3) -> Optional[str]:
        if not self.vector_db:
            raise ValueError("Vector store is not built. Call process_documents() first.")

        results = self.vector_db.similarity_search(query, k=top_k)
        context = "\n".join(doc.page_content for doc in results)
        prompt = self.template.format(question=query, context=context)

        try:
            answer = self.provider.generate_text(prompt, temperature=0.3)
        except Exception as exc:
            self.logger.error("QA generation error: %s", exc)
            return None

        return answer

    # Backward-compat alias
    def GenerateAnswer(self, query: str, top_k: int = 3) -> Optional[str]:  # noqa: N802
        return self.generate_answer(query, top_k)