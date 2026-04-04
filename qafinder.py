# pip install sentence-transformers chromadb

import json
import os

print("\r起動中です。。。", end="", flush=True)

from huggingface_hub import logging

# 警告（Warning）以上の重大なエラー以外は表示しないようにする
logging.set_verbosity_error()

import chromadb
from sentence_transformers import SentenceTransformer


class VectorChatBot:
    def __init__(self, jsonl_path, db_path="./my_vectordb"):
        # 1. モデルの読み込み（おすすめの軽量・高精度モデル）
        print("\rモデルを読み込んでいます...（初回は時間がかかります）", end="", flush=True)
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        
        # 2. データベースの準備（フォルダに保存する設定）
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="qa_collection")
        
        # 3. データの登録（DBが空の場合のみ実行）
        if self.collection.count() == 0:
            print("\rデータベースにデータを登録しています...", end="", flush=True)
            self._load_and_index_data(jsonl_path)
        else:
            print(f"\r既存のデータベース（{self.collection.count()}件）を使用します。", end="", flush=True)

    def _load_and_index_data(self, jsonl_path):
        """JSONLを読み込んでベクトル化し、DBに保存する"""
        if not os.path.exists(jsonl_path):
            print(f"\rエラー: {jsonl_path} が見つかりません。", end="", flush=True)
            return

        documents = []
        embeddings = []
        metadatas = []
        ids = []
        questions = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # JSONLの各行は {"question": "質問文", "answer": "回答文"} の形式であることを想定
                try:
                    data = json.loads(line)
                    question = data.get('question')
                    answer = data.get('answer')
                    if not question or not answer:
                        continue
                except json.JSONDecodeError:
                    continue
                
                # E5モデルのルール: 検索対象（DB側）には "passage: " をつける
                questions.append(f"passage: {question}")

                # ベクトル化は後でまとめて行うため、ここでは質問だけをリストに追加
                documents.append(question)
                metadatas.append({"answer": answer})
                ids.append(f"id_{i}")

            # ベクトル化は一括で行う（効率のため）
            embeddings = self.model.encode(
                questions,
                batch_size=32,
                show_progress_bar=True
            )

        # DBへ一括登録
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("\r登録が完了しました。", end="", flush=True)

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
        
        # 距離が0.5より大きい場合
        if closest_distance > 0.5:
            return "関連する情報が見つかりませんでした。"
        
        # 距離が0.25より大きい場合（0.25 < distance <= 0.5）
        elif closest_distance > 0.25:
            # 距離が近い順に答えを列挙する
            responses = []
            responses.append("以下の中に回答はありますか？：")
            for i, (question, answer, distance) in enumerate(zip(
                results['documents'][0],
                [meta['answer'] for meta in results['metadatas'][0]],
                results['distances'][0]
            ), 1):
                responses.append(f"{i}. 【質問】{question}  => 【回答】{answer}  (距離：{distance:.4f})")   # \n   【距離】{distance:.4f}
            return "\n".join(responses)
        
        # 距離が0.25以下の場合
        else:
            answer = results['metadatas'][0][0]['answer']
            return answer + f"  (距離：{closest_distance:.4f})"

# --- 実行セクション ---
if __name__ == "__main__":
    # 事前に dataset.jsonl を用意しておいてください
    bot = VectorChatBot("dataset.jsonl")
    
    print("\nチャットボット準備完了！ (終了するには exit と入力)")
    while True:
        user_input = input("\n質問を入力してください: ")
        if user_input.lower() in ['exit', 'quit', '終了', 'えぃｔ', 'bye', 'びぇ', 'くいｔ']:
            break
            
        response = bot.ask(user_input)
        print(f"ボットからの回答  : {response}")
