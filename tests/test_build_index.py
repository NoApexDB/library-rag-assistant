"""
Тесты для build_index.py — только детерминированные части без сети
(detect_format, parse_fb2). Модель эмбеддингов и Ollama сюда не берём —
это отдельная, более тяжёлая история (нужны сеть/GPU), не для unit-тестов.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_index import collect_books, detect_format, parse_fb2, parse_pdf

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


def test_parse_fb2_extracts_annotation():
    result = parse_fb2(TEST_BOOK)
    assert result["annotation"] == "Сборник рассказов о ведьмаке Геральте."


def test_parse_pdf_extracts_metadata(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({
        "/Title": "Глубокое обучение",
        "/Author": "Ян Гудфеллоу",
        "/Subject": "Учебник по искусственному интеллекту и нейросетям.",
    })
    pdf_path = tmp_path / "deep_learning.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    result = parse_pdf(pdf_path)
    assert result["title"] == "Глубокое обучение"
    assert result["author"] == "Ян Гудфеллоу"
    assert "Учебник по искусственному интеллекту" in result["annotation"]


def test_parse_pdf_fallback_to_filename(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_path = tmp_path / "pattern_recognition_and_ml.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    result = parse_pdf(pdf_path)
    assert result["title"] == "pattern recognition and ml"
    assert result["author"] == ""
    assert result["annotation"] == ""


def test_parse_pdf_corrupted_file_falls_back_gracefully(tmp_path):
    corrupted_pdf = tmp_path / "corrupted_book.pdf"
    corrupted_pdf.write_bytes(b"%PDF-broken-header-junk")

    result = parse_pdf(corrupted_pdf)
    assert result["title"] == "corrupted book"
    assert result["author"] == ""


def test_collect_books_supports_pdf_and_fb2(tmp_path):
    from pypdf import PdfWriter

    # Создаём PDF
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Title": "Test ML PDF", "/Author": "Author ML"})
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Копируем тестовый FB2
    fb2_path = tmp_path / "sample.fb2"
    fb2_path.write_bytes(TEST_BOOK.read_bytes())

    # Мусорный файл
    junk_path = tmp_path / "random.txt"
    junk_path.write_text("Hello world")

    books = collect_books(tmp_path)
    assert len(books) == 2
    titles = {b["title"] for b in books}
    assert "Test ML PDF" in titles
    assert "Последнее желание" in titles

