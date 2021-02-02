import sys
import jieba
from string import punctuation
import re

##去除中英文标点
class Normalizer:
    """
    sentence normalized
    """
    @staticmethod
    def delete_punc(sentences):
        punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
        return [re.sub(r"[{}]+".format(punc), '', s) for s in sentences]
    @staticmethod
    def delete_e_word(sentences):
        ENGLISH_RE = re.compile(r'[a-zA-Z]+')
        return [ENGLISH_RE.sub('', s) for s in sentences]
    @staticmethod
    def delete_special_symbol(sentences):
        SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')
        return [SPECIAL_SYMBOL_RE.sub('', s) for s in sentences]

    @staticmethod
    def filter_empty(sentences):
        return [c for c in sentences if len(c.strip())]
    @staticmethod
    def normalize(sentences):
        sentences = Normalizer.delete_e_word(sentences)
        sentences = Normalizer.delete_punc(sentences)
        sentences = Normalizer.delete_special_symbol(sentences)
        sentences = Normalizer.filter_empty(sentences)
        return sentences

##分词器
class Segmentator:
    """
    sentence segmentator
    """
    @staticmethod
    def seg_sentences(sentences):
        cut_words = jieba.cut(sentences,cut_all=False)
        return [c for c in list(cut_words) if len(c.strip())]

def preprocess(content):
    sentences = Segmentator.seg_sentences(content)
    sentences = Normalizer.normalize(sentences)
    return sentences


def load_train_file(file,line_num=1000):
    cnt = 0
    data = []
    for line in open(file,encoding='gbk'):
        line = line.replace('\n','').strip()
        if not cnt:
            cnt += 1
            continue
        if len(line.split(',')) !=4:
            continue
        if cnt > line_num:
            break
        id,cate,content,num = line.split(',')
        words = preprocess(content)
        data.append(words)
        cnt += 1
    return data

if __name__ == '__main__':
    # sentences = '北京清华大学'
    # print(list(Segmentator.seg_sentences(sentences)))
    # print(preprocess(sentences))
    data = load_train_file('../data/sohu_train.txt',line_num=10)
    print(data[0])