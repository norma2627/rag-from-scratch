# LangChainでEmbedding・ベクトルストアの構築を実装
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# ドキュメント読み込み・チャンク分割
loader = TextLoader('cat-facts.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f'チャンク数: {len(all_splits)}')


# Embeddingモデルのセットアップ
embeddings = OllamaEmbeddings(model='hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')

# 動作確認: 1つのテキストをベクトル化し、時間を計測
start = time.time()
test_vector = embeddings.embed_query('cats are fast')
embed_time = time.time() - start

print('---Embeddingの動作確認---')
print(f'次元数: {len(test_vector)}')
print(f'先頭5次元: {[round(v, 4) for v in test_vector[:5]]}')
print(f'1クエリのEmbedding時間: {embed_time:.3f}秒')


# ベクトルストアの構築（時間測定も行う）
print(f'\n---ベクトルストア構築---')
start = time.time()
vector_store = InMemoryVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
)
store_time = time.time() - start

print(f'{len(all_splits)}チャンクを格納')
print(f'構築時間: {store_time:.2f}秒')
print(f'1チャンクあたり: {store_time/len(all_splits):.3f}秒')


# 類似度検索（時間測定も行う）
query = 'How fast can cats run?'
print(f'\n---類似度検索---')
print(f'Query: "{query}"\n')

# 検索時間の測定
start = time.time()
results = vector_store.similarity_search(query, k=3)
search_time = time.time() - start

print('---検索結果(上位3件)---')
for i, doc in enumerate(results):
    print(f'Chunk {i}: {doc.page_content[:80]}...')
print(f'\n検索時間: {search_time:.3f}秒')

# スコア付き検索
results_with_score = vector_store.similarity_search_with_score(query, k=3)

print(f'\n---スコア付き検索結果---')
for i, (doc, score) in enumerate(results_with_score):
    print(f'Chunk {i}: (score: {score:.4f}) {doc.page_content[:80]}...')


# 複数クエリで比較
queries = [
    'How fast can cats run?',
    'Tell me about cats',
    'What is the meaning of life?',
]

print(f'\n---クエリ別の検索結果・時間比較---')
for query in queries:
    start = time.time()
    results = vector_store.similarity_search_with_score(query, k=3)
    q_time = time.time() - start

    print(f'\nQuery: "{query}" ({q_time:.3f}秒)')
    for i, (doc, score) in enumerate(results):
        print(f'  {i+1}. (score: {score:.4f}) {doc.page_content[:60]}...')


# 処理時間まとめ
print(f'\n---処理時間まとめ---')
print(f'Embedding(1クエリ):      {embed_time:.3f}秒')
print(f'ベクトルストア構築:       {store_time:.2f}秒 ({len(all_splits)}チャンク)')
print(f'類似度検索（1クエリ）:    {search_time:.3f}秒')