import json
import difflib

class SimpleChatBot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.qa_data = self.load_data()

    def load_data(self):
        """JSONLファイルを読み込んでリストに格納する"""
        qa_list = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 1行ずつ辞書形式に変換してリストに追加
                    qa_list.append(json.loads(line))
        except FileNotFoundError:
            print(f"エラー: {self.data_path} が見つかりません。")
        return qa_list

    def find_answer(self, user_query):
        """ユーザーの質問に最も近い回答を探す"""
        best_match = None
        highest_score = 0
        
        for item in self.qa_data:
            # 文字列の類似度を計算 (0.0 〜 1.0)
            score = difflib.SequenceMatcher(None, user_query, item['question']).ratio()
            
            if score > highest_score:
                highest_score = score
                best_match = item['answer']
        
        # 類似度が低すぎる（例: 0.3以下）場合は「わからない」と返す
        if highest_score > 0.3:
            return best_match
        else:
            return "申し訳ありません。その質問に対する回答は見つかりませんでした。"

# --- 実行部分 ---
if __name__ == "__main__":
    # インスタンス作成
    bot = SimpleChatBot("dataset.jsonl")
    
    print("チャットボットを起動しました（終了するには 'exit' と入力してください）")
    
    while True:
        user_input = input("ユーザー: ")
        if user_input.lower() in ['exit', 'quit', '終了']:
            break
            
        answer = bot.find_answer(user_input)
        print(f"ボット: {answer}")