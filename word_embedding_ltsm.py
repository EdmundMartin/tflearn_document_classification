import numpy as np
from collections import Counter
import re
import pickle
import tflearn
from tflearn.data_utils import pad_sequences


class WordEmbeddingLTSM:

    def __init__(self, vocab_size, max_document_length, class_label, train_split=0.1, dimensions=256, learning_rate=0.001):
        self.documents = []
        self.labels = []
        self.max_document_length = max_document_length
        self.vocab_size = vocab_size
        self.class_labels = [label.lower() for label in class_label]
        self.train_split = train_split
        self.vocab = None
        self.model = None
        self.learning_rate = learning_rate
        self.dimensions = dimensions

    def _load_csv(self, csv_file):
        with open(csv_file, 'r') as input_file:
            for line in input_file:
                split_line = line.split(',')
                self.labels.append(split_line[-1].strip())
                self.documents.append(''.join(split_line[:-1]).lower())

    def _tokenize(self, document):
        return re.findall('\w+', document)

    def _build_vocab(self):
        total_counts = Counter()
        for i in range(len(self.documents)):
            tokens = self._tokenize(self.documents[i])
            for token in tokens:
                total_counts[token] += 1
        vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:self.vocab_size]
        word2idx = {}
        for i, word in enumerate(vocab, 1):
            word2idx[word] = i
        self.vocab = word2idx
        save_vocab = open('vocab.pkl', "wb")
        pickle.dump(word2idx, save_vocab)
        save_vocab.close()

    def _labels_to_nums(self, label):
        zeros = [0 for i in range(len(self.class_labels))]
        index = self.class_labels.index(label.lower())
        zeros[index] = 1
        return zeros

    def _prepare_data(self):
        X = []
        for document in self.documents:
            word_ids = np.zeros(self.max_document_length, np.int64)
            doc_tokens = self._tokenize(document)
            for idx, token in enumerate(doc_tokens):
                if idx >= self.max_document_length:
                    break
                word_id = self.vocab.get(token)
                if word_id is None:
                    word_ids[idx] = 0
                else:
                    word_ids[idx] = word_id
            X.append(word_ids)
        X = pad_sequences(X, maxlen=self.max_document_length, value=0.)
        y = [self._labels_to_nums(label) for label in self.labels]
        y = [np.array(label) for label in y]

        split =  int(len(X) * self.train_split)
        X_train, X_test = X[split:], X[:split]
        y_train, y_test = y[split:], y[:split]
        return X_train, X_test, y_train, y_test

    def _build_model(self):
        net = tflearn.input_data([None, self.max_document_length])
        net = tflearn.embedding(net, input_dim=self.vocab_size+1, output_dim=self.dimensions)
        net = tflearn.lstm(net, self.dimensions, dropout=0.8)
        net = tflearn.fully_connected(net, len(self.class_labels), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=self.learning_rate,
                                 loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        return model

    def _prepare_document(self, document):
        word_ids = np.zeros(self.max_document_length, np.int64)
        doc_tokens = self._tokenize(document)
        for idx, token in enumerate(doc_tokens):
            if idx >= self.max_document_length:
                break
            word_id = self.vocab.get(token)
            if word_id is None:
                word_ids[idx] = 0
            else:
                word_ids[idx] = word_id
        return word_ids

    def train_model(self, csv_file, model_name, epochs=5, batch_size=32):
        self._load_csv(csv_file)
        self._build_vocab()
        X_train, X_test, y_train, y_test = self._prepare_data()
        model = self._build_model()
        model.fit(X_train, y_train, n_epoch=epochs, shuffle=True, validation_set=(X_test, y_test), show_metric=True, batch_size=batch_size)
        model.save(model_name)
        self.model = model

    def load_model(self, model_file, vocab_file):
        voacab = open(vocab_file, "rb")
        self.vocab = pickle.load(voacab)
        voacab.close()
        model = self._build_model()
        model.load(model_file)
        self.model = model

    def predict_document(self, document):
        document_data = self._prepare_document(document)
        result = self.model.predict([document_data])[0]
        most_probable = max(result)
        results = list(result)
        most_probable_index = results.index(most_probable)
        class_name = self.class_labels[most_probable_index]
        return class_name, results


if __name__ == '__main__':
    w = WordEmbeddingLTSM(vocab_size=5000, max_document_length=500, class_label=['Positive', 'Negative'],
                     train_split=0.1)
    #import os
    #model_file = os.path.join(os.getcwd(), 'new_model')
    #print(model_file)
    #w.load_model(model_file, 'vocab.pkl')
    w.train_model('example.csv', 'new_model_ltsm')
    #res = w.predict_document('Most boring tv show ever. What a joke so bad')