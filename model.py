import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class Model(object):
    def build(self,
              features, # This is batch_features from input_fn
              labels,   # This is batch_labels from input_fn
              mode,     # An instance of tf.estimator.ModeKeys
              params):

        self.input_data = features['x']
        self.params = params

        if mode == tf.estimator.ModeKeys.TRAIN: self.dropout = 0.5
        else: self.dropout = 1.

        self.max_over_time_pooled_layers = []
        with tf.name_scope("cnn-layer"):
            self.max_over_time_pooled_layer = self._cnn_layer(self.input_data, params['kernels'])
            self.max_over_time_pooled_layers.append(self.max_over_time_pooled_layer)

        with tf.name_scope("concatenated-layer"):
            self.concatenated_layer = tf.concat(self.max_over_time_pooled_layers, axis=1)

        with tf.name_scope('fc-layer'):
            self.highway_layer = self._fc_layer(input_layer=self.concatenated_layer, \
                                                   training=(mode==tf.estimator.ModeKeys.TRAIN))

        # logits
        with tf.name_scope('logit-layer'):
            self.logits_layer = self._logits_layer(self.highway_layer, params['num_classes'])


        # hypothesis
        with tf.name_scope('hypothesis'):
            self.hypothesis = tf.nn.softmax(self.logits_layer, name='hypothesis')

        # evel
        if mode == tf.estimator.ModeKeys.PREDICT:
            with tf.name_scope('eval'):
                self.prob, self.answer = tf.nn.top_k(self.hypothesis, 1)
            predictions = {
#                 'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': self.prob,
                'predict_index': self.answer,
                'hypothesis': self.hypothesis,
                'grad_cam': self.get_grad_cam()
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # train
        with tf.name_scope("train"):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits_layer)
            self.optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(self.loss, \
                                                                                      global_step=tf.train.get_global_step())

        # accuracy
        with tf.name_scope('accuracy'):
            self.acc, self.acc_update_op = tf.metrics.accuracy(labels=labels, \
                                               predictions=tf.argmax(self.logits_layer, axis=1), \
                                               name='acc_op')

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, \
                                              eval_metric_ops={'accuracy': (self.acc, self.acc_update_op)})

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({'accuracy':self.acc}, every_n_iter=100)
            train_op = tf.group(self.optimizer, self.acc_update_op)
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, \
                                              train_op=train_op, training_hooks=[logging_hook])



    def _fc_layer(self, input_layer, training):
        layer = tf.layers.dense(input_layer, units=100,
                                activation=tf.nn.relu, \
                                kernel_initializer=xavier_initializer(), \
                                bias_initializer=tf.zeros_initializer())
        layer = tf.layers.dropout(layer, rate=0.5, training=training)
        layer = tf.layers.dense(layer, units=50, \
                                activation=tf.nn.relu, \
                                kernel_initializer=xavier_initializer(), \
                                bias_initializer=tf.zeros_initializer())
        layer = tf.layers.dropout(layer, rate=0.5, training=training)
        return layer

    def _cnn_layer(self, x, kernels, name_scope_postfix=""):
        self.feature_maps = []
        pooled_outputs = []
        for i, (filter_size, filter_num) in enumerate(kernels):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # x shape: [None, length, embedding_dim]
                conv = tf.layers.conv1d(
                    inputs=x,
                    filters=filter_num,
                    kernel_size=filter_size,
                    strides=1,
                    padding="valid",
                    activation=tf.nn.relu)
                # conv shape: [None, length-filter_size+1, filter_num]
                self.feature_maps.append(conv)
                pooled = tf.layers.average_pooling1d(
                    inputs=conv,
                    pool_size=int(conv.get_shape()[-2]),
                    strides=1)
                # pooled shape: [None, 1, filter_num]
                pooled_outputs.append(tf.squeeze(pooled, axis=[1]))

        # Combine all the pooled features
        flatten = tf.concat(values=pooled_outputs, axis=1)
        return flatten

    def _logits_layer(self, x, num_classes):
        logits = tf.layers.dense(x, units=num_classes,
                                 activation=None, \
                                 kernel_initializer=xavier_initializer(), \
                                 bias_initializer=tf.zeros_initializer())
        return logits

    def get_grad_cam(self, class_idxs=[]):
        if len(class_idxs)==0:
            class_idxs=list(range(self.params['num_classes']))

        grad_cam = []
        for _class_idx in class_idxs:

            y_c = self.logits_layer[:,_class_idx]
            grad_cam_c_filtersize = []
            for feature_map in self.feature_maps:
                # shape: [None, length-filter_size+1, filter_num]

                _dy_da = tf.gradients(y_c, feature_map)[0]
                # shape: [None, length-filter_size+1, filter_num]

                _alpha_c = tf.reduce_mean(_dy_da, axis=1)
                # shape: [None, filter_num]

                _grad_cam_c = tf.nn.relu(tf.reduce_sum(tf.multiply(tf.transpose(feature_map, perm=[0,2,1]),
                                                                   tf.stack([_alpha_c], axis=2)),
                                                       axis=1))
                # L_gradcam_c = relu(sigma(alpha*feature_map))   (broadcasting multiply)
                # shape: [None, length-filter_size+1]

                _interpol_grad_cam_c = tf.stack([tf.stack([_grad_cam_c], axis=2)], axis=3)
                _interpol_grad_cam_c = tf.image.resize_bilinear(images=_interpol_grad_cam_c, size=[self.params['max_article_length'],1])
                _interpol_grad_cam_c = tf.squeeze(_interpol_grad_cam_c, axis=[2,3])
                # shape: [None, length]

                grad_cam_c_filtersize.append(_interpol_grad_cam_c)

            grad_cam_c = tf.reduce_sum(tf.stack(grad_cam_c_filtersize, axis=0), axis=0)
            # grad_cam_c shape: [None, length]    (element wise sum for each grad cam per filter_size)
            grad_cam_c = grad_cam_c / tf.norm(grad_cam_c, axis=1, keepdims=True)
            # grad_cam_c shape: [None, length]    (element wise normalize)

            grad_cam.append(grad_cam_c)

        return tf.stack(grad_cam, axis=1)
        # shape: [None, num_classes, length]
