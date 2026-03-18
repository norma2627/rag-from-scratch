# LangChainを使用したドキュメントの読み込み

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DocumentLoaderを使用したドキュメントの読み込み
loader = TextLoader('cat-facts.txt')
docs = loader.load()

print("---ドキュメントの読み込み---")
print(f'ドキュメント数; {len(docs)}')
print(f'文字数: {len(docs[0].page_content)}')
print(f'メタデータ: {docs[0].metadata}')
print(f'先頭200文字: \n{docs[0].page_content[:200]}')

# RecursiveCharacterTextspplitterを用いたチャンク分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50, # 隣り合うチャンク同士で重複させる文字(チャンク付近にある情報が失われるのを防ぐ) 
    add_start_index = True,
)

all_splits = text_splitter.split_documents(docs)

print('---チャンク分割結果---')
print(f'チャンク数: {len(all_splits)}')
print()

# 先頭5チャンクの内容を確認
print('---先頭5チャンク')
for i, split in enumerate(all_splits[:3]):
    content = split.page_content
    print(f'Chunk {i}:')
    print(f'先頭: "{content[:50]}..."')
    print(f'末尾: "...{content[-50:]}"')
    print()


# スクラッチ実装との比較（メモ）
print('---スクラッチ実装との比較---')
print(f'スクラッチ（1行=1チャンク）: 150チャンク')
print(f'スクラッチ（固定長200文字）: 150チャンク')
print(f'LangChain（RecursiveCharacterTextsplitter）: {len(all_splits)}チャンク')
print()