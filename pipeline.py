# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from config import config
from retriever import BookRetriever
from llm import OllamaClient

SYSTEM_PROMPT = """Ты — эрудированный библиотекарь и персональный книжный ментор (Library RAG Assistant).
Твоя задача: помочь читателю найти идеальные книги под его запрос, настроение или задачу.

ПРАВИЛА:
1. Тебе предоставлен список реально найденных книг из каталога библиотеки с метаданными.
2. Опирайся на найденные книги и дай структурированный, увлекательный ответ.
3. Объясни, ПОЧЕМУ каждая из рекомендованных книг точно попадает в запрос читателя.
4. Укажи точного автора, название и файл архива для каждой книги.
5. Пиши живым, вовлекающим и профессиональным языком.
"""

class LibraryRAG:
    def __init__(self):
        self.retriever = BookRetriever()
        self.llm = OllamaClient()

    def ask(self, user_query: str) -> Dict[str, Any]:
        # 1. Этап поиска кандидатов
        candidates = self.retriever.retrieve(user_query, top_k=config.final_top_k)
        
        if not candidates:
            return {
                "answer": "К сожалению, по вашему запросу не удалось найти подходящих книг в каталоге.",
                "candidates": []
            }
            
        # 2. Формирование контекста для LLM
        books_context = "\n".join([
            f"- [{i+1}] Автор: {b.get('author', 'Неизвестен')} | Название: «{b.get('title', 'Без названия')}» "
            f"(Файл: {b.get('zip_file', '')} / {b.get('file', '')}, Семантическое сходство: {b.get('score', 0):.3f})"
            for i, b in enumerate(candidates)
        ])
        
        user_prompt = f"""Запрос читателя: "{user_query}"

Найденные кандидаты из библиотеки:
{books_context}

Сформируй персональный ответ читателю с разбором и рекомендациями по лучшим из этих книг:"""

        # 3. Генерация ответа через LLM (или fallback)
        if self.llm.is_available():
            answer = self.llm.generate(user_prompt, system_prompt=SYSTEM_PROMPT)
        else:
            # Fallback режим, если Ollama не запущена
            answer = (
                "[Ollama не запущена / оффлайн режим]: Отображаются сырые семантические результаты:\n\n" +
                "\n".join([f"{i+1}. {b['author']} — «{b['title']}» (Сходство: {b['score']:.3f})" for i, b in enumerate(candidates)])
            )
            
        return {
            "query": user_query,
            "answer": answer,
            "candidates": candidates
        }
