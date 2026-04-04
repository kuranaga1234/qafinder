import json
from janome.tokenizer import Tokenizer

class NLPChatBot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenizer = Tokenizer()
        self.qa_data = self.load_data()

    def load_data(self):
        qa_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 学習データの質問文もあらかじめ単語に分解しておく（高速化のため）
                item['tokens'] = self.get_nouns(item['question'])
                qa_list.append(item)
        return qa_list

    def get_nouns(self, text):
        """文章から名詞だけを抽出する"""
        return [token.surface for token in self.tokenizer.tokenize(text) 
                if token.part_of_speech.startswith('名詞')]

    def find_answer(self, user_query):
        user_tokens = set(self.get_nouns(user_query))
        
        best_match = None
        max_overlap = 0

        for item in self.qa_data:
            # ユーザーの質問単語と、データの質問単語がどれだけ重なっているか
            overlap = len(user_tokens.intersection(set(item['tokens'])))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = item['answer']
        
        if max_overlap > 0:
            return best_match
        return "すみません、そのキーワードに関する情報はありません。"

# 実行
bot = NLPChatBot("dataset.jsonl")
print(bot.find_answer("お店の営業時間はいつですか？"))