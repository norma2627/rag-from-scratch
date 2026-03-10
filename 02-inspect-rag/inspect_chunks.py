"""
チャンクの中身を見る
- 各チャンクの内容と文字数を表示
- チャンクの長さの分布を可視化
- 固定長分割との比較
"""

import matplotlib.pyplot as plt


# データセットの読み込み
dataset = [] 
with open('cat-facts.txt', 'r') as file:
    dataset = [line.strip() for line in file.readlines() if line.strip()]

# entriesは行数を数える
print(f'Loaded {len(dataset)} entries\n')


# 各チャンクの内容と文字数を表示
print('--- 各のチャンクの内容（先頭5件）--- ')
for i, chunk in enumerate(dataset[:5]):
    print(f'Chunk {i}: ({len(chunk)} chars) {chunk[:80]}...') # 先頭80文字を表示
print()

# チャンクの長さの分布を可視化
chunk_lengths = [len(chunk) for chunk in dataset]

plt.figure(figsize=(10, 4))
plt.bar(range(len(chunk_lengths)), chunk_lengths, color='skyblue')
plt.xlabel('Chunk Index')
plt.ylabel('Character Count')
plt.title('Length of Each Chunk')
plt.tight_layout()
plt.savefig('./png/chunk_lengths.png', dpi=150)
plt.show()

print(f'最短: {min(chunk_lengths)} chars')
print(f'最長: {max(chunk_lengths)} chars')
print(f'平均: {sum(chunk_lengths)/len(chunk_lengths):.0f} chars')
print()

# 固定長分割との比較
def chunk_by_fixed_length(text, chunk_size=200, overlap=50):
    """固定長でチャンクに分割する"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

full_text = ' '.join(dataset)
fixed_chunks = chunk_by_fixed_length(full_text, chunk_size=200, overlap=50)

print(f'元のチャンク数: {len(dataset)}')
print(f'固定長分割チャンク数: {len(fixed_chunks)}')
print(f'\n--- 固定チャンクの例（先頭3件） ---')
for i, chunk in enumerate(fixed_chunks[:3]):
    print(f'Chunk {i}: ({len(chunk)} chars) {chunk}')
    print()