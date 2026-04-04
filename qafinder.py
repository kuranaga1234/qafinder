# pip install sentence-transformers chromadb

import json
import os
import random

print("\r起動中です。。。", end="", flush=True)

from huggingface_hub import logging

# 警告（Warning）以上の重大なエラー以外は表示しないようにする
logging.set_verbosity_error()

import chromadb
from sentence_transformers import SentenceTransformer


# カテゴリ一覧
CATEGORIES = ["人事", "通勤", "給与", "その他"]

# カテゴリ判定キーワード辞書
CATEGORY_KEYWORDS = {
    "人事": [
        "人事", "評価", "等級", "降格", "昇格", "昇進", "制度", "職種", "スペシャリスト",
        "マネジメント", "ランク", "資格手当", "新卒", "コアバリュー", "市場価値",
        "スキル", "採用", "入社", "退職", "規定", "問い合わせ", "フィードバック",
    ],
    "通勤": [
        "通勤", "交通", "定期", "定期券", "バス", "電車", "鉄道", "バス停", "経路",
        "らくらく", "らくらくBOSS", "申請", "承認", "代理", "手当", "実費",
        "徒歩", "紛失", "割引", "障害者", "2km", "2Km",
    ],
    "給与": [
        "給与", "給料", "賞与", "ボーナス", "基本給", "昇給", "減給", "手当",
        "業績手当", "所得税", "非課税", "単価", "チャージ", "月給", "季節賞与",
        "決算賞与", "支給", "報酬",
    ],
}


# ===== 案3: 挨拶ルール辞書 =====
# キーワードをキー、回答候補リストを値とする辞書。
# キーワードはqueryへの部分一致で判定する。
GREETING_RESPONSES: dict[str, list[str]] = {
    "こんにちは": ["良い天気ですね。", "こんにちは。", "ご機嫌はいかがですか。",
                   "こんにちは！社内QAについてお答えします。何かお手伝いしましょうか？"],
    "こんばんは": ["こんばんは。", "夜遅くまでお疲れ様です。", "ご機嫌はいかがですか。"],
    "おはよう":   ["おはようございます。", "今日も一日頑張りましょう！", "良い朝ですね。"],
    "はじめまして": ["はじめまして！社内QAについてお答えします。",
                     "よろしくお願いします！何かお手伝いしましょうか？"],
    "ありがとう": ["どういたしまして！", "お役に立てて嬉しいです。",
                   "またいつでも聞いてくださいね。", "少しでもお力になれて良かったです。"],
    "お疲れ様":   ["お疲れ様です！", "今日もお疲れ様でした。",
                   "ゆっくり休んでくださいね。", "お疲れ様です！何かサポートできることはありますか？"],
    "よろしく":   ["こちらこそよろしくお願いします！", "何でも聞いてください。"],
    "さようなら": ["またいつでもどうぞ。", "お疲れ様でした！", "ご利用ありがとうございました。"],
    "バイバイ":   ["またいつでもどうぞ。", "お疲れ様でした！"],
}


def try_greeting(query: str) -> str | None:
    """
    案3: クエリが挨拶キーワードにマッチすれば候補からランダムに1つ返す。
    マッチしなければ None を返してベクトル検索へ進む。
    """
    for keyword, responses in GREETING_RESPONSES.items():
        if keyword in query:
            return random.choice(responses)
    return None


def pick_answer(raw_answer: str) -> str:
    """
    案1: answerが'|'区切りの複数候補ならランダムに1つ選ぶ。
    単一回答ならそのまま返す。
    """
    candidates = [s.strip() for s in raw_answer.split("|") if s.strip()]
    return random.choice(candidates) if candidates else raw_answer


def detect_categories(query: str) -> list[str]:
    """
    クエリのキーワードから検索対象カテゴリを判定する。
    複数カテゴリにマッチする場合は複数返す。
    どれにもマッチしない場合は全カテゴリを返す（全体検索）。
    """
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            matched.append(category)

    # 「その他」は明示マッチしないので、未マッチ時のフォールバックとして全カテゴリ検索
    if not matched:
        return CATEGORIES  # 全カテゴリを検索
    return matched


def collection_name(category: str) -> str:
    """カテゴリ名をChromaDBのコレクション名に変換する（英数字のみ使用）"""
    mapping = {
        "人事": "qa_jinji",
        "通勤": "qa_tsukin",
        "給与": "qa_kyuyo",
        "その他": "qa_sonota",
    }
    return mapping[category]


class VectorChatBot:
    def __init__(self, jsonl_path, db_path="./my_vectordb"):
        # 1. モデルの読み込み
        print("\rモデルを読み込んでいます...（初回は時間がかかります）", end="", flush=True)
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')

        # 2. データベースの準備
        self.client = chromadb.PersistentClient(path=db_path)

        # 3. カテゴリごとにコレクションを作成・取得
        self.collections = {}
        for category in CATEGORIES:
            self.collections[category] = self.client.get_or_create_collection(
                name=collection_name(category)
            )

        # 4. いずれかのコレクションが空なら全データを再登録
        any_empty = any(col.count() == 0 for col in self.collections.values())
        if any_empty:
            print("\rデータベースにデータを登録しています...", end="", flush=True)
            self._load_and_index_data(jsonl_path)
        else:
            counts = {cat: col.count() for cat, col in self.collections.items()}
            summary = "、".join(f"{cat}:{n}件" for cat, n in counts.items())
            print(f"\r既存のデータベース（{summary}）を使用します。", end="", flush=True)

    def _load_and_index_data(self, jsonl_path):
        """JSONLを読み込んでカテゴリ別にベクトル化し、DBに保存する"""
        if not os.path.exists(jsonl_path):
            print(f"\rエラー: {jsonl_path} が見つかりません。")
            return

        # カテゴリ別にデータを仕分け
        data_by_category: dict[str, dict] = {
            cat: {"questions": [], "documents": [], "metadatas": [], "ids": []}
            for cat in CATEGORIES
        }

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            global_id = 0
            for line in f:
                try:
                    data = json.loads(line)
                    question = data.get('question')
                    answer = data.get('answer')
                    category = data.get('category', 'その他')
                    if not question or not answer:
                        continue
                    if category not in CATEGORIES:
                        category = 'その他'
                except json.JSONDecodeError:
                    continue

                bucket = data_by_category[category]
                bucket["questions"].append(f"passage: {question}")
                bucket["documents"].append(question)
                bucket["metadatas"].append({"answer": answer, "category": category})
                bucket["ids"].append(f"id_{global_id}")
                global_id += 1

        # カテゴリごとにエンコード＆登録
        for category, bucket in data_by_category.items():
            if not bucket["questions"]:
                continue
            embeddings = self.model.encode(
                bucket["questions"],
                batch_size=32,
                show_progress_bar=False,
            ).tolist()
            self.collections[category].add(
                documents=bucket["documents"],
                embeddings=embeddings,
                metadatas=bucket["metadatas"],
                ids=bucket["ids"],
            )
            print(f"\r  [{category}] {len(bucket['documents'])}件 登録完了", flush=True)

        print("\rすべてのカテゴリの登録が完了しました。", end="", flush=True)

    def ask(self, query: str) -> str:
        """ユーザーの質問に対してカテゴリを絞り込んで回答を検索する"""
        # ── 案3: 挨拶ルール辞書で先に捌く ──────────────────────────
        greeting = try_greeting(query)
        if greeting is not None:
            print("  ※ 検索カテゴリ: ルール辞書（挨拶）")
            return greeting
        # ────────────────────────────────────────────────────────────

        # 検索対象カテゴリを判定
        target_categories = detect_categories(query)
        category_label = "・".join(target_categories)
        print(f"  ※ 検索カテゴリ: {category_label}")

        # E5モデルのルール: 質問側には "query: " をつける
        query_vector = self.model.encode([f"query: {query}"]).tolist()

        # 対象カテゴリのコレクションをまとめて検索し、結果をフラット化
        all_results = []  # (distance, question, answer, category)

        for category in target_categories:
            col = self.collections[category]
            if col.count() == 0:
                continue
            results = col.query(
                query_embeddings=query_vector,
                n_results=min(3, col.count()),
            )
            if not results['ids'][0]:
                continue
            for question, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            ):
                all_results.append((distance, question, meta['answer'], meta.get('category', category)))

        if not all_results:
            return "すみません、答えが見つかりませんでした。"

        # 距離でソートして最良の結果を取得
        all_results.sort(key=lambda x: x[0])
        closest_distance = all_results[0][0]

        # 距離が0.5より大きい場合 → 関連情報なし
        if closest_distance > 0.5:
            return "関連する情報が見つかりませんでした。"

        # 距離が0.25より大きい場合 → 候補を列挙
        elif closest_distance > 0.25:
            responses = ["以下の中に回答はありますか？："]
            for i, (distance, question, answer, cat) in enumerate(all_results[:3], 1):
                # 案1: |区切り複数候補があればランダム選択
                responses.append(
                    f"{i}. 【{cat}】【質問】{question}  => 【回答】{pick_answer(answer)}  (距離：{distance:.4f})"
                )
            return "\n".join(responses)

        # 距離が0.25以下 → 最良の回答を返す
        else:
            _, _, answer, cat = all_results[0]
            # 案1: |区切り複数候補があればランダム選択
            return f"【{cat}】{pick_answer(answer)}  (距離：{closest_distance:.4f})"


# --- 実行セクション ---
if __name__ == "__main__":
    bot = VectorChatBot("dataset.jsonl")

    print("\nチャットボット準備完了！ (終了するには exit と入力)")
    while True:
        user_input = input("\n質問を入力してください: ")
        if user_input.lower() in ['exit', 'quit', '終了', 'えぃｔ', 'bye', 'びぇ', 'くいｔ']:
            break

        response = bot.ask(user_input)
        print(f"ボットからの回答  : {response}")
