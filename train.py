import tensorflow as tf
from dataset import *
import os
import configargparse
from model import Model
import time

def train(epoch, batch_size, learning_rate, max_article_length):
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model = Model()
    classifier = tf.estimator.Estimator(model_fn=model.build,
                                        config=tf.estimator.RunConfig(session_config=config,
                                                                      model_dir=os.path.join('ckpt', time.strftime("%m%d_%H%M%S"))),
                                        params={
                                            'feature_columns': [tf.feature_column.numeric_column(key='x')], \
                                            'kernels': [(3,512),(4,512),(5,512)], \
                                            'num_classes': 2, \
                                            'learning_rate': learning_rate, \
                                            'max_article_length': max_article_length
    })
    
    data = SST(Word2vecEnWordEmbedder)
    classifier.train(input_fn=lambda: data.train_input_fn(batch_size=batch_size, padded_size=max_article_length, epoch=epoch))
    eval_val = classifier.evaluate(input_fn=lambda: data.eval_input_fn(batch_size=batch_size, padded_size=max_article_length))
    print("----------------Evaluation Test Set----------------")
    print(eval_val)
    
    
if __name__=="__main__":
    MAX_ARTICLE_LENGTH = 500
    
    parser = configargparse.ArgParser()
    parser.add("--epoch", dest="epoch", help="Train Epoch", default=10, type=int)
    parser.add("--batch-size", dest="batch_size", help="Train Batch Size", default=300, type=int)
    parser.add("--learning-rate", dest="learning_rate", help="Train Learning Rate", default=0.001, type=float)
    parser.add("--gpu-index", dest="gpu_index", help="GPU Index Number", default="0", type=str)

    args = vars(parser.parse_args())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_index']

    train(epoch=args['epoch'],
          batch_size=args['batch_size'],
          learning_rate=args['learning_rate'],
          max_article_length=MAX_ARTICLE_LENGTH)