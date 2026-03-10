"""
ベクトルの中身を見る
- 埋め込みベクトルの値を確認
- 似た文と違う文のベクトルを比較
- PCA / t-SNEで可視化
"""

import ollama
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

# コサイン類似度の算出
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


# データセットの読み込み
dataset = []
with open('cat-facts.txt', 'r') as file:
    dataset = [line.strip() for line in file.readlines() if line.strip()]

print(f'Loaded {len(dataset)} entries\n')


# 埋め込みベクトル値を確認
sample_text = dataset[0]
embedding = ollama.embed(model=EMBEDDING_MODEL, input=sample_text)['embeddings'][0]

print('--- 埋め込みベクトルの中身 ---')
print(f'テキスト: {sample_text[:60]}...')
print(f'ベクトルの次元数: {len(embedding)}') # ベクトルモデルの次元数
print(f'先頭10次元の値: {[round(v, 4) for v in embedding[:10]]}')
print(f'値の範囲: {min(embedding):.4f} ~ {max(embedding):.4f}')
print(f'平均: {np.mean(embedding):.4f}')
print(f'標準偏差: {np.std(embedding):.4f}')
print()


# 似た文章同士の比較
texts = [
    "cats are very fast animals",
    "felines can run at high speed",
    "the global economy is growing rapidly",
]

embeddings_compare = []
for text in texts:
    emb = ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]
    embeddings_compare.append(emb)

print('--- コサイン類似度の比較 ---')
print(f'"cats are fast" VS "felines run": {cosine_similarity(embeddings_compare[0], embeddings_compare[1]):.4f}')
print(f'"cats are fast" VS "economy is growing": {cosine_similarity(embeddings_compare[0], embeddings_compare[2]):.4f}')
print(f'"felines run" VS "economy is growing": {cosine_similarity(embeddings_compare[1], embeddings_compare[2]):.4f}')
print()


# 全チャンクの埋め込みを取得
print('全チャンクの埋め込みを取得中...')
VECTOR_DB = []
for i, chunk in enumerate(dataset):
    emb = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, emb))
    print(f' {i+1}/{len(dataset)}', end='\r')
print(f'\n完了: {len(VECTOR_DB)} チャンク\n')


# PCAで可視化
all_embeddings = np.array([emb for _, emb in VECTOR_DB])

pca = PCA(n_components=2)
reduced_2d = pca.fit_transform(all_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c='steelblue', s=100, alpha=0.7)

for i, (chunk, _) in enumerate(VECTOR_DB):
    plt.annotate(
        chunk[:20] + '...',
        (reduced_2d[i, 0], reduced_2d[i, 1]),
        fontsize=7, alpha=0.8,
        textcoords="offset points", xytext=(5, 5),
    )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Chunk embedding visualized with PCA(2D)')
plt.tight_layout()
plt.savefig('./png/pca_scatter.png', dpi=150)
plt.show()
print('PCA散布図を保存しました: pca_scatter.png\n')


# t-SNEで可視化
perplexity = min(5, len(dataset) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
reduced_tsne = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c='coral', s=100, alpha=0.7)

for i, (chunk, _) in enumerate(VECTOR_DB):
    plt.annotate(
        chunk[:20] + '...',
        (reduced_tsne[i, 0], reduced_tsne[i, 1]), fontsize=7, alpha=0.8,
        textcoords="offset points", xytext=(5,5),
    )

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Chunk embeddings visualized with t-SNE')
plt.tight_layout()
plt.savefig('./png/tsne_scatter.png', dpi=150)
plt.show()
print('t-SNE散布図を保存しました: tsne_scatter.png')
