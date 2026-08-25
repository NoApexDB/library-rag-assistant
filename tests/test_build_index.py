"""
Тесты для build_index.py — только детерминированные части без сети
(detect_format, parse_fb2). Модель эмбеддингов и Ollama сюда не берём —
это отдельная, более тяжёлая история (нужны сеть/GPU), не для unit-тестов.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_index import detect_format, parse_fb2

TEST_BOOK = Path(__file__).resolve().parent / "test_book.fb2"


def test_detect_format_recognizes_fb2():
    assert detect_format(TEST_BOOK) == "fb2"


def test_detect_format_recognizes_pdf(tmp_path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%rest of file...")
    assert detect_format(fake_pdf) == "pdf"


def test_detect_format_unknown_extension_but_real_fb2_content(tmp_path):
    """Тот самый случай с .a4.pdf — расширение врёт, но по содержимому определяем верно."""
    tricky_file = tmp_path / "book.a4"
    tricky_file.write_bytes(TEST_BOOK.read_bytes())
    assert detect_format(tricky_file) == "fb2"


def test_detect_format_returns_none_for_garbage(tmp_path):
    garbage = tmp_path / "not_a_book.bin"
    garbage.write_bytes(b"\x00\x01\x02random binary junk")
    assert detect_format(garbage) is None


def test_parse_fb2_extracts_title_and_author():
    result = parse_fb2(TEST_BOOK)
    assert result["title"] == "Последнее желание"
    assert result["author"] == "Анджей Сапковский"


def test_parse_fb2_missing_title_falls_back_gracefully(tmp_path):
    """Файл без book-title вообще — не должен падать, должен вернуть заглушку."""
    minimal_fb2 = tmp_path / "no_title.fb2"
    minimal_fb2.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        '<description><title-info></title-info></description>'
        '</FictionBook>',
        encoding="utf-8",
    )
    result = parse_fb2(minimal_fb2)
    assert result["title"] == "Без названия"
    assert result["author"] == ""
