"""
類似度スコアの中身を見る
- 全チャンクとの類似度を一覧表示
- 類似度の分布をグラフで可視化
- クエリを変えて分布を比較
"""

import ollama
import matplotlib.pyplot as plt

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


# データセット読み込み + ベクトルDB構築
dataset = []
with open('cat-facts.txt', 'r')as file:
    dataset = [line.strip() for line in file.readlines() if line.strip()]

print(f'Loaded {len(dataset)} entries')

print('ベクトルDB構築中...')
VECTOR_DB = []
for i, chunk in enumerate(dataset):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))
    print(f' {i+1}/{len(dataset)}', end='\r')
print(f'\n完了\n')


# 全チャンクとの類似度を一覧表示
def retrieve_all(query):
    "全チャンクとの類似度を返す"
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

query = "How fast can cats run?"
all_results = retrieve_all(query)

print(f'Query: "{query}"\n')
print('--- 全チャンクの類似度ランキング ---')
for i, (chunk, sim) in enumerate(all_results):
    marker = '◀ TOP3 ' if i < 3 else ''
    print(f'{i+1:2d}. (similarity: {sim:.4f}) {chunk[:60]}...{marker}')
print()


# 類似度の分布をグラフで可視化
scores = [sim for _, sim in all_results]

plt.figure(figsize=(12, 5))
plt.bar(
    range(len(scores)), scores, color=['#e74c3c' if i < 3 else 'steelblue' for i in range(len(scores))],
)
plt.xlabel('Chunk rank')
plt.ylabel('Cosine similarity')
plt.title(f'Similarity scores for query: "{query}"')
plt.axhline(
    y=scores[2], color='red', linestyle='--', alpha=0.5, label=f'Top3 threshold: {scores[2]:.4f}',
)
plt.legend()
plt.tight_layout()
plt.savefig('./png/similarity_distribution.png', dpi=150)
plt.show()
print('類似度分布を保存しました: similarity_distribution.png\n')


# クエリを変えて比較
queries = [
    "How fast can cats run?",
    "Tell me about cats",
    "What is the meaning of life?",
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, q in zip(axes, queries):
    results = retrieve_all(q)
    q_scores = [sim for _, sim in results]

    ax.bar(
        range(len(q_scores)), q_scores, color=['#e74c3c' if i < 3 else 'steelblue' for i in range(len(q_scores))],
    )
    ax.set_title(f'"{q[:30]}..."', fontsize=10)
    ax.set_xlabel('chunk rank')
    ax.set_ylabel('Cosine similarity')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('./png/query_comparison.png', dpi=150)
plt.show()
print('クエリ比較を保存しました: query_comparsion.png')