from rouge_score import rouge_scorer
from konlpy.tag import Komoran
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

class KoTokenizer:
    def __init__(self):
        self.tok = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        # self.k = Komoran()

    def tokenize(self, sentence):
        # return self.k.morphs(sentence)
        return self.tok(sentence)

class RougeScorer:
    def __init__(self):
        tokenizer = KoTokenizer()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=tokenizer, use_stemmer=True)

        #print(tokenizer.tokenize("너무재밋었다그래서보는것을추천한다"))

    def score(self, s1, s2):
        scores = self.scorer.score(s1, s2)
        return scores

if __name__ == '__main__':
    r = RougeScorer()
    sc = r.score(
        "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.",
        "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.",
    )

    print(sc)
    sc = r.score(
        "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.",
        "형태소 맞춤법이 잘 지켜져있다면, 띄어쓰기가 안 되어 있어도 대체로 잘 지켜지는 모습을 보입니다",
    )
    print(sc)