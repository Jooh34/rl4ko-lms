# -*- coding: utf-8 -*-
from rouge_score import rouge_scorer
from rouge_score import scoring
# from konlpy.tag import Komoran
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu

class MyScore:
    def __init__(self, p, r, f):
        self.precision=p
        self.recall=r
        self.fmeasure=f

    def __str__(self):
        return f"precision: {self.precision}, recall: {self.recall}, fmeasure: {self.fmeasure}"
        
    # adding two objects
    def __add__(self, other):
        precision = self.precision + other.precision
        recall = self.recall + other.recall
        fmeasure = self.fmeasure + other.fmeasure
        return MyScore(precision, recall, fmeasure)

    # divide by value
    def divide_by(self, value):
        self.precision /= float(value)
        self.recall /= float(value)
        self.fmeasure /= float(value)

    
    # multiply by value
    def multiply_by(self, value):
        self.precision *= float(value)
        self.recall *= float(value)
        self.fmeasure *= float(value)

class FakeTokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentence):
        return sentence

class KoTokenizer:
    def __init__(self):
        self.tok = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        # self.k = Komoran()

    def tokenize(self, sentence):
        # return self.k.morphs(sentence)
        return self.tok.encode(sentence)

class RougeScorer:
    def __init__(self, tokenizer=KoTokenizer()):
        self.rouge_types = ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, tokenizer=tokenizer, use_stemmer=True)

        #print(tokenizer.tokenize("너무재밋었다그래서보는것을추천한다"))

    def score(self, s1, s2):
        #list input
        if (not isinstance(s1, str)) and isinstance(s1[0], str):
            if len(s1) != len(s2):
                return None
            
            length = len(s1)
            sum_score = {'rouge1':MyScore(0, 0, 0), 'rouge2':MyScore(0, 0, 0), 'rougeL':MyScore(0, 0, 0)}
            for ix in range(length):
                score = self.scorer.score(s1[ix], s2[ix])
                for rtype in self.rouge_types:
                    sum_score[rtype].precision += score[rtype].precision
                    sum_score[rtype].recall += score[rtype].recall
                    sum_score[rtype].fmeasure += score[rtype].fmeasure
                
            
            for rtype in self.rouge_types:
                sum_score[rtype].precision /= float(length)
                sum_score[rtype].recall /= float(length)
                sum_score[rtype].fmeasure /= float(length)
            
            return sum_score

        else:
            score = self.scorer.score(s1, s2)
            my_score = {'rouge1':MyScore(0, 0, 0), 'rouge2':MyScore(0, 0, 0), 'rougeL':MyScore(0, 0, 0)}
            for rtype in self.rouge_types:
                my_score[rtype].precision = score[rtype].precision
                my_score[rtype].recall = score[rtype].recall
                my_score[rtype].fmeasure = score[rtype].fmeasure

            return my_score

if __name__ == '__main__':
    # reference = [
    #     'this is a dog'.split(),
    #     'it is dog'.split(),
    #     'dog it is'.split(),
    #     'a dog, it is'.split() 
    # ]
    # candidate = 'it is a dog'.split()

    # print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights=(0.33,0.33,0.33,0))))

    # label = " \
    #     8일 서울에서 열린 5G플러스 전략발표에 참석한 문재인 대통령은 5G는 대한민국 혁신성장의 인프라이자 넓고, 체증 없는 '통신 고속도로'라고 강조하며 5G가 각 분양에 융합되면 정보통신산업을 넘어 제조업과 벤처에 이르러 우리 산업 전체의 혁신을 통한 동반성장이 가능하다고 언급했다. \
    # "

    # s1 = "\
    #     '대한민국 5G 홍보대사'를 자처한 문재인 대통령은 8일 서울 올림픽공원에서 열린 5G플러스 전략발표에 참석해 '5G 시대는 우리가 생각하고, 만들면 그것이 세계 표준이 되는 시대'라며 '5G는 대한민국 혁신성장의 인프라라고 강조했다.'\
    # "

    # s2 = "\
    #     문재인 대통령은 8일 서울 올림픽공원에서 열린 5G플러스 전략발표에 참석해 5G가 4차 산업혁명 시대의 고속도로가 돼 새로운 기회를 열어 줄 것이라며, 5G가 4차 산업혁명 시대의 고속도로가 돼 우리 산업 전체의 혁신을 통한 동반성장이 가능하다고 밝혔다.\
    # "

    label = "지난 3일 한국이 세계 첫 5세대 이동통신 서비스를 보편화한 것을 축하하는 '코리안 5G 테크-콘서트'가 개최됐고 이 공연은 5G 이동통신의 실시간 전송기술로 서울·부산·광주에서 함께하는 원격협연 형식으로 진행됐으며 이날 문재인 대통령 및 홍남기 부총리 등 정부 관계자와 업체 대표 등 300여 명이 모여 5G 기술에 찬사를 하였다."
    s1 = "지난 3일 한국이 세계 최초로 5세대(5G) 이동통신 서비스를 상용화한 것을 기념하는 '코리안 5G 테크-콘서트'가 열려 서울과 부산, 광주에서 동시에 하는 원격협연 방식으로 진행되어 마치 한 장소에 모여 연주하고, 연주에 맞춰 탈춤공연을 하는 것 같았다."
    s2 = "지난 3일 한국이 세계 최초로 5세대(5G) 이동통신 서비스를 상용화한 것을 기념하는 '코리안 5G 테크-콘서트'가 열렸는데, 이 공연은 서울과 부산, 광주에서 동시에 하는 원격협연 방식으로 이루어졌으며, SK텔레콤의 원격 협연은 초고속 초저지연 초연결이라는 5G 기술의 3대 특징 중 하나인 초저지연을 실감하게 했다."

    r = RougeScorer()
    sc = r.score(
        label, s1
    )
    print(sc["rouge1"])
    print(sc["rouge2"])
    print(sc["rougeL"])

    # sc = r.score(
    #     ['먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.', "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다."],
    #     ["먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.", "형태소 맞춤법이 잘 지켜져있다면, 띄어쓰기가 안 되어 있어도 대체로 잘 지켜지는 모습을 보입니다"]
    # )

    sc = r.score(
        label, s2
    )
    print(sc["rouge1"])
    print(sc["rouge2"])
    print(sc["rougeL"])
    # sc = r.score(
    #     "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.",
    #     "형태소 맞춤법이 잘 지켜져있다면, 띄어쓰기가 안 되어 있어도 대체로 잘 지켜지는 모습을 보입니다",
    # )
    # print(sc)
    
    # sc = r.score(
    #     "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.",
    #     "먼저, 맞춤법은 잘 지켜졌으나, 띄어쓰기가 하나도 안 되어 있는 경우입니다.",
    # )
    # print(sc)