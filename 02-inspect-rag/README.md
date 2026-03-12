# 02: チャンク・ベクトル・類似度の中身を確認する

前回の実装（01-simple-rag）を元に、RAG の内部の値を確認し理解を深める。

📝対応記事: [【RAG入門②】RAGのチャンク・ベクトル・類似度の中身を覗く](https://zenn.dev/norma/articles/c75659ec2e8561)

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `inspect_chunks.py` | チャンクの分割結果と長さ分布の可視化 |
| `inspect_vectors.py` | 埋め込みベクトルの中身確認と PCA/t-SNEでの可視化 |
| `inspect_similarity.py` | 類似度スコアの全体分布と複数クエリでの比較 |
| `my_rag.py` | 自分のデータで動かす |
| `cat-facts.txt` | ナレッジベース（猫の豆知識） |
| `my-knowledge.txt` | 自分で用意したナレッジベースの例 |

## 実行方法

```bash
# 各スクリプトを個別に実行
python inspect_chunks.py
python inspect_vectors.py
python inspect_similarity.py
python my_rag.py
```

各スクリプトを実行すると、グラフ画像（`.png`）が生成され、pngフォルダ内に保存される。