"""
build_index.py — обходит library_root, извлекает метаданные из каждой книги,
кодирует в эмбеддинги через SentenceTransformer, сохраняет как .pkl индекс.

Архитектура: PARSERS — это словарь "расширение файла -> функция-парсер".
Чтобы добавить новый формат (epub, pdf...) — пишешь отдельную parse_* функцию
и одну строку в PARSERS. collect_books() трогать не нужно вообще.
"""
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import config

FB2_NS = {"fb2": "http://www.gribuser.ru/xml/fictionbook/2.0"}


def parse_fb2(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    title_element = root.find('.//fb2:book-title', FB2_NS)
    title = title_element.text if title_element is not None else "Без названия"

    author_element = root.find('.//fb2:author', FB2_NS)
    final_author = ""
    if author_element is not None:
        last_name_el = author_element.find('fb2:last-name', FB2_NS)
        first_name_el = author_element.find('fb2:first-name', FB2_NS)
        last_name = last_name_el.text if last_name_el is not None else ""
        first_name = first_name_el.text if first_name_el is not None else ""
        final_author = f"{first_name} {last_name}".strip()

    return {"title": title, "author": final_author}


# Другие форматы — добавляешь по мере необходимости, не все сразу:
# def parse_epub(path: Path) -> dict: ...
# def parse_pdf(path: Path) -> dict: ...

PARSERS = {
    ".fb2": parse_fb2,
    # ".epub": parse_epub,
    # ".pdf": parse_pdf,
}


def detect_format(path: Path) -> str | None:
    """Определяет реальный тип файла по содержимому, не по расширению
    (напоминание: расширение может врать, как было с .a4.pdf)."""
    try:
        with open(path, "rb") as f:
            header = f.read(20)
    except Exception:
        return None

    if header.startswith(b"%PDF-"):
        return "pdf"
    if b"<?xml" in header or b"<FictionBook" in header:
        return "fb2"
    return None


def collect_books(library_root: Path) -> list[dict]:
    """Обходит все файлы в library_root, парсит поддерживаемые форматы,
    молча пропускает неподдерживаемые (не падает на них)."""
    books = []
    for path in library_root.rglob("*"):
        if not path.is_file():
            continue

        real_format = detect_format(path)
        parser = PARSERS.get(f".{real_format}") if real_format else None
        if parser is None:
            continue

        try:
            book = parser(path)
            book["file"] = str(path.relative_to(library_root))
            books.append(book)
        except Exception as e:
            print(f"[skip] {path.name}: {e}")
    return books


def build_index():
    print(f"[build_index] Сканирую {config.library_root} ...")
    books = collect_books(config.library_root)
    print(f"[build_index] Найдено книг: {len(books)}")

    if not books:
        print("[build_index] Ничего не найдено — проверь LIBRARY_ROOT и наличие поддерживаемых форматов.")
        return

    print(f"[build_index] Загружаю модель {config.embedding_model_name} ...")
    model = SentenceTransformer(config.embedding_model_name, device=config.embedding_device)

    texts = [f"{b.get('title', '')} {b.get('author', '')} {b.get('annotation', '')}" for b in books]
    embeddings = model.encode(texts, show_progress_bar=True)

    with open(config.index_path, "wb") as f:
        pickle.dump({"books": books, "embeddings": embeddings}, f)

    print(f"[build_index] Индекс сохранён: {config.index_path} ({len(books)} книг)")


if __name__ == "__main__":
    build_index()