import gensim.models as word2vec
from konlpy.tag import Mecab
import jamotools
import tensorflow as tf
import numpy as np
from dataset import Word2vecEmbedder, Dataset

class Word2vecKoMorphEmbedder(Word2vecEmbedder):
    def __init__(self):
        model = word2vec.Word2Vec.load('./word2vec/word2vec_news_morph_300.model')
        max_vocab_size = 20000
        embedding_dim = 300
        super().__init__(model, max_vocab_size, embedding_dim)
        self.pos_tagger = Mecab()

    def get_embedding(self, sentence):
        try:
            embedding = []
            for morph, pos in self.pos_tagger.pos(sentence):
                pair = '/'.join([morph, pos])
                vocab_idx = self.vocab_dict.get(pair, -1)
                if vocab_idx > 0:
                    embedding.append(self.w2v.wv.vectors[vocab_idx])
                else:
                    embedding.append(self.oov)
            return np.array(embedding)
        except:
            # fail pos tagging
            return np.array([self.oov])

class JamoEmbedder():
    def __init__(self):
        self.vectorizationer = jamotools.Vectorizationer(
            rule=jamotools.rules.RULE_1,
            max_length=None)
        self.embedding_dim = len(self.vectorizationer.symbols)

    def _encode_one_hot(self, idxs):
        values = np.array(idxs)
        return np.eye(self.embedding_dim)[values]

    def get_embedding(self, sentence):
        idxs = self.vectorizationer.vectorize(sentence)
        return self._encode_one_hot(idxs)

    
class NSMC(Dataset):
    def __init__(self, embed_cls):
        super().__init__()
        self.train_x, self.train_y = self._load_data('https://raw.githubusercontent.com/HaebinShin/nsmc/master/ratings_train.txt')
        self.test_x, self.test_y = self._load_data('https://raw.githubusercontent.com/HaebinShin/nsmc/master/ratings_test.txt')
        self.embedder = embed_cls()

    def _maybe_download(self, _url):
        _path = tf.keras.utils.get_file(fname=_url.split('/')[-1], origin=_url)
        return _path

    def _load_data(self, url):
        path = self._maybe_download(url)
        contents = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                _id, docu, label = line.split('\t')
                if len(docu.strip())==0: continue
                contents.append(docu.strip())
                labels.append(int(label.strip()))
        return contents, labels

def test():
    data = NSMC(Word2vecKoMorphEmbedder)
    # data = NSMC(JamoEmbedder)
    train_dataset = data.train_input_fn(50, padded_size=100)
    # predict_dataset = data.predict_input_fn(['한국'], padded_size=50)
    # print(predict_dataset)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        features, labels = sess.run(train_dataset)
        print(features['x'].shape)
        print(np.array(labels).shape)

if __name__=="__main__":
    test()