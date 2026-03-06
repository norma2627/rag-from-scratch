# rag-from-scratch

フレームワークを使わず、素の Python + Ollama で RAG を段階的に実装しながら仕組みを理解することを目的としています。

各実装は Zenn の記事と対応しています。


## 実装一覧

| # | フォルダ | 内容 | 記事 |
|---|---------|------|------|
| 01 | [`01-simple-rag`](./01-simple-rag/) | RAG の基本実装（Indexing → Retrieval → Generation） | [【RAG入門】PythonとOllamaでゼロからRAGを実装して仕組みを理解する](https://zenn.dev/norma/articles/cee3bcf1472e00) |
| 02 | [`02-inspect-rag`](./02-inspect-rag/) | チャンク・ベクトル・類似度の中身を覗いて理解を深める | （準備中） |

## 技術スタック

- **Python**（フレームワークなし）
- **Ollama** — ローカル LLM 実行環境
- **埋め込みモデル** — `bge-base-en-v1.5`
- **言語モデル** — `Llama-3.2-1B-Instruct`

> 埋め込みモデルと言語モデルは実装が進むにつれ変更される場合があります。


## セットアップ

```bash
# Ollama のインストール（公式サイト参照: https://ollama.com）

# モデルのダウンロード
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

# Python パッケージ（全実装の依存をまとめてインストール）
pip install ollama matplotlib scikit-learn numpy
```


## ライセンス

MIT