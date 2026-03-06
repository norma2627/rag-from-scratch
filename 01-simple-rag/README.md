# 01: シンプルな RAG の実装

RAGの3フェーズ（Indexing → Retrieval → Generation）を素のPythonで実装する。

📝対応記事: [【RAG入門】PythonとOllamaでゼロからRAGを実装して仕組みを理解する](https://zenn.dev/norma/articles/cee3bcf1472e00)

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `demo.py` | RAG の全体実装 |
| `cat-facts.txt` | ナレッジベース（猫の豆知識） |

## 実行方法

```bash
# Ollama が起動していることを確認
python demo.py
```