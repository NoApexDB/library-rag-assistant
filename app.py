# -*- coding: utf-8 -*-
import sys
from pipeline import LibraryRAG

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def main():
    print("=" * 60)
    print("🧠 Intelligent Library RAG Assistant (700k+ Books)")
    print("=" * 60)
    
    rag = LibraryRAG()
    print("\n[Готово] Ассистент запущен. Введите ваш сложный запрос или 'выход'.")
    
    while True:
        try:
            query = input("\n📖 Ваш запрос: ").strip()
            if not query:
                continue
            if query.lower() in ["выход", "exit", "quit", "q"]:
                print("До встречи!")
                break
                
            print("\n🔎 Ищем подходящие книги и формируем ответ...")
            result = rag.ask(query)
            
            print("\n" + "=" * 60)
            print("🤖 Ответ Библиотекаря:")
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\nСессия завершена.")
            break

if __name__ == "__main__":
    main()
