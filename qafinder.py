# pip install sentence-transformers chromadb

import json
import os
import random
import re
import unicodedata

print("\r起動中です。。。", end="", flush=True)

from huggingface_hub import logging

# 警告（Warning）以上の重大なエラー以外は表示しないようにする
logging.set_verbosity_error()

import chromadb
from rapidfuzz import fuzz
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
# ゆれ表現（表記ゆれ・省略形）も代表キーワードと同じ候補を共有する。
GREETING_RESPONSES: dict[str, list[str]] = {
    # ── こんにちは系 ──────────────────────────────────────────
    "こんにちは":   ["良い天気ですね。", "こんにちは。", "ご機嫌はいかがですか。",
                     "こんにちは！社内QAについてお答えします。何かお手伝いしましょうか？"],
    "こんにちわ":   ["良い天気ですね。", "こんにちは。", "ご機嫌はいかがですか。",
                     "こんにちは！社内QAについてお答えします。何かお手伝いしましょうか？"],
    "ちわ":         ["こんにちは。", "ご機嫌はいかがですか。"],
    # ── こんばんは系 ─────────────────────────────────────────
    "こんばんは":   ["こんばんは。", "夜遅くまでお疲れ様です。", "ご機嫌はいかがですか。"],
    "こんばんわ":   ["こんばんは。", "夜遅くまでお疲れ様です。", "ご機嫌はいかがですか。"],
    # ── おはよう系 ───────────────────────────────────────────
    "おはよう":     ["おはようございます。", "今日も一日頑張りましょう！", "良い朝ですね。"],
    "おはす":       ["おはようございます。", "今日も一日頑張りましょう！"],
    "おは":         ["おはようございます。", "良い朝ですね。"],
    # ── はじめまして系 ───────────────────────────────────────
    "はじめまして": ["はじめまして！社内QAについてお答えします。",
                     "よろしくお願いします！何かお手伝いしましょうか？"],
    "はじめて":     ["はじめまして！社内QAについてお答えします。",
                     "よろしくお願いします！何かお手伝いしましょうか？"],
    # ── ありがとう系 ─────────────────────────────────────────
    "ありがとう":   ["どういたしまして！", "お役に立てて嬉しいです。",
                     "またいつでも聞いてくださいね。", "少しでもお力になれて良かったです。"],
    "ありがと":     ["どういたしまして！", "お役に立てて嬉しいです。",
                     "またいつでも聞いてくださいね。"],
    "サンキュ":     ["どういたしまして！", "お役に立てて嬉しいです。"],
    "感謝":         ["どういたしまして！", "お役に立てて嬉しいです。"],
    # ── お疲れ様系 ───────────────────────────────────────────
    "お疲れ様":     ["お疲れ様です！", "今日もお疲れ様でした。",
                     "ゆっくり休んでくださいね。", "お疲れ様です！何かサポートできることはありますか？"],
    "おつかれ":     ["お疲れ様です！", "今日もお疲れ様でした。", "ゆっくり休んでくださいね。"],
    "おつ":         ["お疲れ様です！", "今日もお疲れ様でした。"],
    # ── よろしく系 ───────────────────────────────────────────
    "よろしく":     ["こちらこそよろしくお願いします！", "何でも聞いてください。"],
    "よろ":         ["こちらこそよろしくお願いします！", "何でも聞いてください。"],
    # ── さようなら系 ─────────────────────────────────────────
    "さようなら":   ["またいつでもどうぞ。", "お疲れ様でした！", "ご利用ありがとうございました。"],
    "さよなら":     ["またいつでもどうぞ。", "お疲れ様でした！", "ご利用ありがとうございました。"],
    "バイバイ":     ["またいつでもどうぞ。", "お疲れ様でした！"],
    "またね":       ["またいつでもどうぞ。", "お疲れ様でした！"],
}

# 案B: fuzzy matching の閾値（0〜100、高いほど厳格）
# 短いキーワード（≤4文字）は誤マッチしやすいため低めの閾値は使わない
FUZZY_THRESHOLD = 85


def normalize(text: str) -> str:
    """
    入力を正規化して表記ゆれを吸収する。
      1. NFKC正規化（全角英数→半角、カタカナはそのまま）
      2. カタカナ→ひらがな変換
      3. 小文字化
      4. 記号・空白・長音符・繰り返し記号を除去
    """
    text = unicodedata.normalize("NFKC", text)   # 全角英数→半角
    # カタカナ（U+30A1〜U+30F6）→ひらがな（U+3041〜U+3096）
    text = "".join(
        chr(ord(ch) - 0x60) if "\u30a1" <= ch <= "\u30f6" else ch
        for ch in text
    )
    text = text.lower()                           # 大文字→小文字
    text = re.sub(r"[！!。、〜～ー\-・\s]", "", text)  # 記号・空白除去
    return text


def try_greeting(query: str) -> str | None:
    """
    挨拶キーワードにマッチすれば候補からランダムに1つ返す。
    マッチしなければ None を返してベクトル検索へ進む。

    判定は2段階:
      ・正規化後のキーワードが正規化後のクエリに部分一致するか
      ・案Aで外れた場合、rapidfuzz の partial_ratio で類似度判定
        ただし短いキーワード（正規化後3文字以下）は誤マッチ防止のため案Bをスキップ

    共通ガード:
      正規化後のクエリ長が「マッチしたキーワードの正規化長 × 2.5」を超える場合は
      挨拶に業務質問が混在していると判断してスキップする。
    """
    norm_query = normalize(query)

    # --- 案A: 正規化キーワードの部分一致 ---
    for keyword, responses in GREETING_RESPONSES.items():
        norm_kw = normalize(keyword)
        if norm_kw in norm_query:
            # ガード: クエリが長すぎる場合は業務質問が混在していると判断してスキップ
            if len(norm_query) > len(norm_kw) * 2.5:
                continue
            print(f"  ※ 挨拶判定（正規化マッチ） キーワード=「{keyword}」")
            return random.choice(responses)

    # --- 案B: fuzzy matching（案Aで外れた場合のフォールバック） ---
    best_score = 0
    best_responses: list[str] | None = None
    best_keyword = ""
    for keyword, responses in GREETING_RESPONSES.items():
        norm_kw = normalize(keyword)
        # 短いキーワードはfuzzy対象外（誤マッチ防止）
        if len(norm_kw) <= 3:
            continue
        score = fuzz.partial_ratio(norm_kw, norm_query)
        if score > best_score:
            best_score = score
            best_responses = responses
            best_keyword = keyword

    if best_responses and best_score >= FUZZY_THRESHOLD:
        norm_best_kw = normalize(best_keyword)
        # ガード: クエリが長すぎる場合はスキップ
        if len(norm_query) <= len(norm_best_kw) * 2.5:
            print(f"  ※ 挨拶判定（fuzzyマッチ） キーワード=「{best_keyword}」 スコア={best_score}")
            return random.choice(best_responses)

    return None


def pick_answer(raw_answer: str) -> str:
    """
    answerが'|'区切りの複数候補ならランダムに1つ選ぶ。
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
