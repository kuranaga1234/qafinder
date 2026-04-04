# pip install sentence-transformers chromadb

import json
import os

import chromadb
from chromadb.config import Settings
from huggingface_hub import logging

from sentence_transformers import SentenceTransformer

# 警告（Warning）以上の重大なエラー以外は表示しないようにする
logging.set_verbosity_error()

class VectorChatBot:
    def __init__(self, jsonl_path, db_path="./my_vectordb"):
        # 1. モデルの読み込み（おすすめの軽量・高精度モデル）
        print("モデルを読み込んでいます...（初回は時間がかかります）")
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        
        # 2. データベースの準備（フォルダに保存する設定）
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="qa_collection")
        
        # 3. データの登録（DBが空の場合のみ実行）
        if self.collection.count() == 0:
            print("データベースにデータを登録しています...")
            self._load_and_index_data(jsonl_path)
        else:
            print(f"既存のデータベース（{self.collection.count()}件）を使用します。")

    def _load_and_index_data(self, jsonl_path):
        """JSONLを読み込んでベクトル化し、DBに保存する"""
        if not os.path.exists(jsonl_path):
            print(f"エラー: {jsonl_path} が見つかりません。")
            return

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                question = data['question']
                answer = data['answer']
                
                # E5モデルのルール: 検索対象（DB側）には "passage: " をつける
                encoded_text = self.model.encode(f"passage: {question}")
                
                documents.append(question)
                embeddings.append(encoded_text.tolist())
                metadatas.append({"answer": answer})
                ids.append(f"id_{i}")

        # DBへ一括登録
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("登録が完了しました。")

    def ask(self, query):
        """ユーザーの質問に対して回答を検索する"""
        # E5モデルのルール: 質問側には "query: " をつける
        query_vector = self.model.encode([f"query: {query}"]).tolist()
        
        # 似ているデータを3件まで取得
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=3
        )

        if not results['ids'][0]:
            return "すみません、答えが見つかりませんでした。"

        # 最も距離が近い結果の距離を確認
        closest_distance = results['distances'][0][0]
        
        # 距離が0.6より大きい場合
        if closest_distance > 0.6:
            return "関連する情報が見つかりませんでした。"
        
        # 距離が0.4より大きい場合（0.3 < distance <= 0.6）
        elif closest_distance > 0.3:
            # 距離が近い順に答えを列挙する
            responses = []
            responses.append("以下の中に回答はありますか？：")
            for i, (question, answer, distance) in enumerate(zip(
                results['documents'][0],
                [meta['answer'] for meta in results['metadatas'][0]],
                results['distances'][0]
            ), 1):
                responses.append(f"{i}. 【質問】{question}  => 【回答】{answer}")   # \n   【距離】{distance:.4f}
            return "\n".join(responses)
        
        # 距離が0.3以下の場合
        else:
            answer = results['metadatas'][0][0]['answer']
            return answer

# --- 実行セクション ---
if __name__ == "__main__":
    # 事前に dataset.jsonl を用意しておいてください
    bot = VectorChatBot("dataset.jsonl")
    
    print("\nチャットボット準備完了！ (終了するには exit と入力)")
    while True:
        user_input = input("\nユーザー: ")
        if user_input.lower() in ['exit', 'quit', '終了']:
            break
            
        response = bot.ask(user_input)
        print(f"ボット  : {response}")
