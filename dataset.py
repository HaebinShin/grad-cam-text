import gensim.models as word2vec
import tensorflow as tf
import numpy as np

class Word2vecEmbedder():
    def __init__(self, model, max_vocab_size, embedding_dim):
        self.w2v = model
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self._build()

    def _build(self):
        self.vocab_dict = self._get_vocab_dict(self.w2v, self.max_vocab_size)
        self.oov = [0 for _ in range(self.embedding_dim)]

    def _get_vocab_dict(self, model, max_vocab_size):
        assert model != None, "word2vec was not trained."
        vocab_dict = {}
        for key in sorted(model.wv.vocab):
            if model.wv.vocab[key].__dict__['index']<max_vocab_size :
                vocab_dict[key] = model.wv.vocab[key].__dict__['index']
        return vocab_dict

    def get_embedding(self, sentence):
        raise NotImplementedError


class Word2vecEnWordEmbedder(Word2vecEmbedder):
    def __init__(self):
        model = word2vec.KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300-SLIM.bin', \
                                                           binary=True)
        max_vocab_size = 20000
        embedding_dim = 300
        super().__init__(model, max_vocab_size, embedding_dim)

    def get_embedding(self, sentence):
        embedding = []
        for word in sentence.split(' '):
            vocab_idx = self.vocab_dict.get(word, -1)
            if vocab_idx > 0:
                embedding.append(self.w2v.wv.vectors[vocab_idx])
            else:
                embedding.append(self.oov)
        return np.array(embedding)

    
class Dataset():
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.embedder = None
    
    def _generator(self, _x, _y=None):
        def _internal_generator():
            for idx, line in enumerate(_x):
                sentence_feature = self.embedder.get_embedding(line)
                if _y == None:
                    yield (sentence_feature, -1)
                else:
                    yield (sentence_feature, _y[idx])
        return _internal_generator
    
    def train_input_fn(self, batch_size, padded_size, epoch=20, shuffle=True):
        g = self._generator(self.train_x, self.train_y)
        dataset = tf.data.Dataset.from_generator(g, output_types=(tf.float32, tf.int32),
                                                 output_shapes=([None, self.embedder.embedding_dim], []))
        if shuffle:
            dataset = dataset.shuffle(9876543)
        dataset = dataset.repeat(epoch)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([padded_size, self.embedder.embedding_dim], []))
        iterator = dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()
        return {"x": feature}, label

    def eval_input_fn(self, batch_size, padded_size):
        g = self._generator(self.test_x, self.test_y)
        dataset = tf.data.Dataset.from_generator(g, output_types=(tf.float32, tf.int32),
                                                 output_shapes=([None, self.embedder.embedding_dim], []))
        dataset = dataset.padded_batch(batch_size, padded_shapes=([padded_size, self.embedder.embedding_dim], []))
        iterator = dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()
        return {"x": feature}, label

    def predict_input_fn(self, _inputs: list, padded_size):
        g = self._generator(_inputs)
        dataset = tf.data.Dataset.from_generator(g, output_types=(tf.float32, tf.int32),
                                                 output_shapes=([None, self.embedder.embedding_dim], []))
        dataset = dataset.padded_batch(len(_inputs), padded_shapes=([padded_size, self.embedder.embedding_dim], []))
        iterator = dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()
        return {"x": feature}
    
class SST(Dataset):
    def __init__(self, embed_cls):
        super().__init__()
        self.train_x, self.train_y = self._load_data('https://raw.githubusercontent.com/HaebinShin/stanford-sentiment-dataset/master/stsa.binary.phrases.train')
        self.test_x, self.test_y = self._load_data('https://raw.githubusercontent.com/HaebinShin/stanford-sentiment-dataset/master/stsa.binary.test')
        self.embedder = embed_cls()

    def _maybe_download(self, _url):
        _path = tf.keras.utils.get_file(fname=_url.split('/')[-1], origin=_url)
        return _path

    def _load_data(self, url):
        path = self._maybe_download(url)
        contents = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line[0]
                docu = line[2:]
                if len(docu.strip())==0: continue
                contents.append(docu.strip())
                labels.append(int(label.strip()))
        return contents, labels
