# -*- coding: utf-8 -*-
import pickle
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import config

class BookRetriever:
    """
    Векторный ретривер для быстрого поиска кандидатов по 700k+ книг.
    """
    def __init__(self, index_path: str = config.index_path, device: str = config.embedding_device):
        print(f"[Retriever] Загрузка векторного индекса из {index_path}...")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            
        self.books: List[Dict[str, Any]] = data["books"]
        self.embeddings: np.ndarray = data["embeddings"]
        
        # Предварительно считаем нормы для мгновенного косинусного сходства
        self.books_norms: np.ndarray = np.linalg.norm(self.embeddings, axis=1)
        print(f"[Retriever] Загружено {len(self.books)} книг. Размерность векторов: {self.embeddings.shape[1]}")
        
        print(f"[Retriever] Инициализация модели {config.embedding_model_name} на {device}...")
        self.model = SentenceTransformer(config.embedding_model_name, device=device)

    def retrieve(self, query: str, top_k: int = config.top_k_candidates) -> List[Dict[str, Any]]:
        """
        Кодирует запрос и возвращает топ-K уникальных кандидатов с косинусным скором.
        """
        # 1. Вектор запроса
        query_vec = self.model.encode(query)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
            
        # 2. Векторизованный расчёт косинусного сходства
        dot_product = np.dot(self.embeddings, query_vec)
        scores = dot_product / (self.books_norms * query_norm)
        
        # 3. Сортировка по убыванию
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        seen = set()
        
        for idx in top_indices:
            book = self.books[idx]
            key = (book.get("author", "").strip().lower(), book.get("title", "").strip().lower())
            
            if key in seen or not book.get("title"):
                continue
                
            seen.add(key)
            book_entry = dict(book)
            book_entry["score"] = float(scores[idx])
            results.append(book_entry)
            
            if len(results) >= top_k:
                break
                
        return results
