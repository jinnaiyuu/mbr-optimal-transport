import MeCab
from evaluate import load

from utility.utility_class import UtilityFunction


class MeCabTokenizer:
    def __init__(self, use_stemmer=False):
        self._stemmer = use_stemmer

        self.tagger = MeCab.Tagger()
        self.wakati = MeCab.Tagger("-Owakati")

    def tokenize(self, text):
        if self._stemmer:
            node = self.tagger.parseToNode(text)
            original_forms = []
            while node:
                feature = node.feature.split(",")
                original_forms.append(feature[6])
                node = node.next

            return original_forms

        else:
            return self.wakati.parse(text).split()


class ROUGELJA(UtilityFunction):
    def __init__(self):
        self.similarity = load("rouge")
        self.mecab_tokenizer = MeCabTokenizer()

    def compute_similarity(self, hyp, ref, src):
        # scores = [self.similarity.compute(predictions=[hyp[i]],
        #                                     references=[ref[i]],
        #                                     use_stemmer=False, tokenizer=self.mecab_tokenizer.tokenize)['rougeL']
        #             for i in range(len(hyp))]
        scores = self.similarity.compute(
            predictions=hyp,
            references=ref,
            use_stemmer=False,
            tokenizer=self.mecab_tokenizer.tokenize,
            use_aggregator=False,
        )["rougeL"]
        return scores
