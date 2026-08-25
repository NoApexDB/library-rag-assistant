
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGConfig:
    index_path: Path = Path(os.environ["LIBRARY_INDEX_PATH"])
    library_root: Path = Path(os.environ["LIBRARY_ROOT"])

    # 1. Модели эмбеддинга
    embedding_model_name: str = "cointegrated/rubert-tiny2"
    embedding_device: str = "cpu"  # 'cuda' или 'cpu'
    
    # 2. Параметры поиска
    top_k_candidates: int = 30     # Первичный векторный пул
    final_top_k: int = 5           # Количество книг в контекст LLM
    
    # 3. Настройки Ollama / LLM
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"  # Можно вставить любую модель из Ollama
    llm_temperature: float = 0.3
    

config = RAGConfig()
