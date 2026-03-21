# LangChainでRetriever・QAチェーンを実装
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore


# ---パイプライン構築---
# ドキュメントの読み込み・チャンク分割
loader = TextLoader('cat-facts.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50,
    add_start_index = True,
)
all_splits = text_splitter.split_documents(docs)
print(f'チャンク数: {len(all_splits)}')

# Embedding・ベクトルストア構築
embeddings = OllamaEmbeddings(model='hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')

print('ベクトルストア構築中...')
start = time.time()
vector_store = InMemoryVectorStore.from_documents(
    documents = all_splits,
    embedding = embeddings,
)
store_time = time.time() - start
print(f'構築完了: {store_time:.2f}秒\n')

# LLMのセットアップ
llm = OllamaLLM(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')


# LLM単体とRAGの比較
def ask_llm_only(llm, query):
    start = time.time()
    response = llm.invoke(query)
    gen_time = time.time() - start
    return response, gen_time

def ask_with_rag(llm, vectore_store, query, k=3):
    #検索
    search_start = time.time()
    results = vector_store.similarity_search_with_score(query, k=k)
    search_time = time.time() - search_start

    # プロンプト構築
    context = '\n'.join([f' - {doc.page_content}' for doc, score in results])
    prompt = f'''
    You are a helpful chatbot.
    Use only the following piece of context to answer the question.
    Don't make up any new information:
    {context}

    Question: {query}'''

    # 生成
    gen_start = time.time()
    response = llm.invoke(prompt)
    gen_time = time.time() - gen_start

    return response, results, search_time, gen_time


# ---検証---
queries = [
    'How fast can cats run?',
    'Why do cats sleep so much?',
    'what is the meaning of life?',
]

print('=' * 50)
print('LLM単体とRAGの比較')
print('=' * 50)

for query in queries:
    print(f'\n{"=" * 50}')
    print(f'Query: "{query}"')
    print('=' * 50)

    # LLM単体
    llm_response, llm_time = ask_llm_only(llm, query)

    print(f'\n---LLM単体（生成時間: {llm_time:.2f}秒）---')
    print(llm_response[:300])
    if len(llm_response) > 300:
        print('...(truncated)')

    # RAG
    rag_response, results, search_time, gen_time = ask_with_rag(llm, vector_store, query)

    print(f'\n---RAG（検索: {search_time:.3f} + 生成: {gen_time:.2f}秒 = 合計{search_time + gen_time:.2f}秒)---')
    print('検索されたチャンク: ')
    for i, (doc, score) in enumerate(results):
        print(f' {i+1}.(score: {score:.4f}) {doc.page_content[:60]}...')
    print(f'\n回答')
    print(rag_response[:300])
    if len(rag_response) > 300:
        print('...(truncated)')

    # 時間比較
    print(f'\n---処理時間の比較---')
    print(f' LLM単体: {llm_time:.2f}秒')
    print(f' RAG合計: {search_time + gen_time: .2f}秒（検索: {search_time:.3f}秒 + 生成: {gen_time:.2f}秒）')
    print(f' 差分: {(search_time + gen_time) - llm_time:+.2f}秒')