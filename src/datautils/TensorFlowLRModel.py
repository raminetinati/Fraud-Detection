''''
Author: Ramine Tinati: raminetinati AT gmail dot com
'''
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.metrics import classification_report
from imblearn.over_sampling import BorderlineSMOTE

'''
A simple class to handle A simple low-level TensorFlow Logistic Regression Model
'''
class TensorFlowLRModel:

    def __init__(self, data=None,
                 learning_rate=0.01,
                 training_steps=10000,
                 batch_size=128,
                 display_step=100):

        print('Neural Network Loading')

        self.model_data = data

        X_train, X_test, y_train, y_test = self.model_data

        self.num_classes = len(np.unique(y_train))  # total classes (0-9 digits).

        self.num_features = X_train.shape[1]  # data features (img shape: 28*28).
        print('Classes {}, Features {}'.format(self.num_classes, self.num_features))

        # Training parameters.

        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.display_step = display_step

        # A random value generator to initialize weights.
        self.random_normal = tf.initializers.RandomNormal()

        # Stochastic gradient descent optimizer.
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        # Network parameters.
        self.n_hidden_1 = 126  # 1st layer number of neurons.
        self.n_hidden_2 = 256  # 2nd layer number of neurons.
        self.n_hidden_3 = 512  # 3rd layer number of neurons.

        self.weights = {
            'h1': tf.Variable(self.random_normal([self.num_features, self.n_hidden_1])),
            'h2': tf.Variable(self.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'h3': tf.Variable(self.random_normal([self.n_hidden_2, self.n_hidden_3])),
            'out': tf.Variable(self.random_normal([self.n_hidden_3, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_1])),
            'b2': tf.Variable(tf.zeros([self.n_hidden_2])),
            'b3': tf.Variable(tf.zeros([self.n_hidden_3])),
            'out': tf.Variable(tf.zeros([self.num_classes]))
        }

    # Create model.
    def neural_net(self, x):
        # Hidden fully connected layer with 128 neurons.
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Apply sigmoid to layer_1 output for non-linearity.
        layer_1 = tf.nn.sigmoid(layer_1)

        # Hidden fully connected layer with 256 neurons.
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Apply sigmoid to layer_2 output for non-linearity.
        layer_2 = tf.nn.sigmoid(layer_2)

        # Hidden fully connected layer with 512 neurons.
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3'])
        # Apply sigmoid to layer_2 output for non-linearity.
        layer_3 = tf.nn.sigmoid(layer_3)

        # Output fully connected layer with a neuron for each class.
        out_layer = tf.matmul(layer_3, self.weights['out']) + self.biases['out']
        # Apply softmax to normalize the logits to a probability distribution.
        return tf.nn.softmax(out_layer)

    # Cross-Entropy loss function.
    def cross_entropy(self, y_pred, y_true):
        # Encode label to a one hot vector.
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        # Clip prediction values to avoid log(0) error.
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
        # Compute cross-entropy.
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

    # Accuracy metric.
    def accuracy(self, y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    def splt(self, y, class_a=0, class_b=1):
        zero = []
        one = []
        for x in y:
            if x == class_a:
                zero.append(0)
            else:
                one.append(1)

        print('Class {}: {} Class {}: {}'.format(class_a, len(zero), class_b, len(one)))

    def make_ds(self, features, labels):
        BUFFER_SIZE = 100000
        ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
        ds = ds.shuffle(BUFFER_SIZE).repeat()
        return ds

    def resample(self, model_data, class_weights):

        X_train, X_test, y_train, y_test = model_data
        y_train = list(y_train)
        X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)
        self.splt(y_train)

        # split data by pos and neg features as we need to balance data
        pos_features = []
        neg_features = []
        pos_labels = []
        neg_labels = []

        for i in range(0, len(X_train)):
            #         print('y i; ',y_train[i])
            if y_train[i] is 0:
                neg_features.append(X_train[i])
                neg_labels.append(y_train[i])
            else:
                pos_features.append(X_train[i])
                pos_labels.append(y_train[i])

        pos_ds = self.make_ds(pos_features, pos_labels)
        neg_ds = self.make_ds(neg_features, neg_labels)
        if class_weights:
            resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=class_weights)
            train_data = resampled_ds.batch(self.batch_size).prefetch(2)

        else:
            train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_data = train_data.repeat().shuffle(X_train.shape[0]).batch(self.batch_size)

        return train_data

    # Optimization process.
    def run_optimization(self, x, y):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            pred = self.neural_net(x)
            loss = self.cross_entropy(pred, y)

        # Variables to update, i.e. trainable variables.
        trainable_variables = list(self.weights.values()) + list(self.biases.values())

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)

        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def run_tf_model(self, class_weights):

        train_data = self.resample(self.model_data, class_weights=class_weights)

        #         # Training parameters.

        # Run training for the given number of steps.
        for step, (batch_x, batch_y) in enumerate(train_data.take(self.training_steps), 1):
            # Run the optimization to update W and b values.
            self.run_optimization(batch_x, batch_y)

            if step % self.display_step == 0:
                self.splt(batch_y)

                pred = self.neural_net(batch_x)
                loss = self.cross_entropy(pred, batch_y)
                acc = self.accuracy(pred, batch_y)
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

    # Test model on validation set.
    def predict(self, model_data, balance_classes=False):

        X_train, X_test, y_train, y_test = model_data
        y_test = list(y_test)

        if balance_classes:
            smote = SMOTE()

            X_test, y_test = smote.fit_resample(X_test, y_test)

        self.splt((y_test))

        X_test = np.array(X_test, np.float32)

        preds = self.neural_net(X_test)

        print("Test Accuracy: %f" % self.accuracy(preds, y_test))

        pred_lbls = tf.argmax(preds, 1)

        return list(pred_lbls.numpy()), y_test