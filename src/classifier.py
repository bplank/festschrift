__author__ = "bplank"

"""
 classifier
"""
import argparse
import os
from itertools import count
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import numpy as np
from collections import Counter
from sklearn.dummy import DummyClassifier
from features import Featurizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
import numpy as np
from sklearn.cross_validation import StratifiedKFold, permutation_test_score

parser = argparse.ArgumentParser()
parser.add_argument('train', help="train data file")
parser.add_argument('dev', help="dev data file")
parser.add_argument('test', help="test data file")
parser.add_argument('--ngrams', help="word ngrams", default=None)
parser.add_argument('--cngrams', help="char ngrams", default=None)
parser.add_argument('--embeds', help="use poly embeddings average", action="store_true", default=False)
parser.add_argument('--output', help='output predictions (gold\t pred)')

args = parser.parse_args()


def evaluate(gold, pred):
    print("")
    print("precision:", precision_score(gold, pred,average='weighted'))
    print("recall:", recall_score(gold, pred,average='weighted'))
    print("f1:", f1_score(gold, pred,average='weighted'))
    print("accuracy:", accuracy_score(gold,pred))
    print("=======")

def load_file(filename, args):
    lines = [l.strip() for l in open(filename).readlines()]
    header = lines[0].strip().split("\t")
    y=[]
    data=[]

    for line in lines[1:]:
        fields = line.split("\t")
        assert(len(header)==len(fields))
        label = int(fields[0]) #user id
        y.append(label)
        ## features
        if "text" not in header[1:]:
            data.append({dict_key: float(i) for dict_key, i in zip(header[1:],fields[1:])})
        else:
            featuredict = {}
            for dict_key, i in zip(header[1:], fields[1:]):
                if not dict_key == "text":
                    featuredict[dict_key] = float(i)
                else:
                    if not args.embeds:
                        f = Featurizer()
                        if args.ngrams:
                            f.word_ngrams(i, ngram=args.ngrams)
                        if args.cngrams:
                            f.character_ngrams(i,ngram=args.cngrams)
                        di =  f.getDict()
                        for k in di:
                            featuredict[k] = di[k]
                    else:
                        words = i.split(" ") # trivial tokenization
                        avg_emb = np.mean([emb.get(w,emb["_UNK"]) for w in words],axis=0)
                        for i,val in enumerate(avg_emb):
                            featuredict["d_{}".format(i)] = val
            data.append(featuredict)
    return data, y

def load_data(data_train, data_dev, data_test, args, vectorizer):
    ### load data
    X_train, y_train = load_file(data_train, args)
    X_dev, y_dev = load_file(data_dev, args)
    X_test, y_test = load_file(data_test, args)
    print(X_train[0])
    ### convert to features
    X_train = vectorizer.fit_transform(X_train)
    X_dev = vectorizer.transform(X_dev)
    X_test = vectorizer.transform(X_test)
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def show_most_informative_features(vectorizer, clf, n=10):
    feature_names = vectorizer.get_feature_names()
    for i in range(0,len(clf.coef_)):
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        print("i",i)
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


def load_embeddings():
    file_name= "/Users/bplank/projects/kerastagger/embeds/poly_a/en.polyglot.txt"
    emb = {}
    for line in open(file_name):
        fields = line.split()
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        emb[word] = vec
    return emb

## read input data
print("load data..")
##
vectorizer = DictVectorizer()

print("load embeddings..")
emb = load_embeddings()

X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.train, args.dev, args.test, args, vectorizer)
print("#train instances: {}\n#test instances: {}\n#dev instances: {}".format(X_train.shape,X_test.shape, X_dev.shape))
assert(X_train.shape[0]==len(y_train))
assert(X_test.shape[0]==len(y_test))
assert(X_dev.shape[0]==len(y_dev))

print("build model")

majority_label = Counter(y_train).most_common()[0][0]
majority_prediction = [majority_label for label in y_test]

#classifier = LogisticRegression()
classifier = SVC(kernel='linear')
print(classifier)
random = DummyClassifier(strategy='stratified')
random.fit(X_train, y_train)

### Q2: add code to train and evaluate your classifier
print("train model..")
## your code here:
classifier.fit(X_train, y_train)
##
print("evaluate model..")
## your code here:
y_test_pred = classifier.predict(X_test)
y_dev_pred = classifier.predict(X_dev)
#y_test_dummy = random.predict(X_test)
#y_dev_dummy = random.predict(X_dev)
###
print("Dev:")
evaluate(y_dev, y_dev_pred)
print("Test:")
evaluate(y_test, y_test_pred)

if args.output:
    OUT=open("{}.dev.out".format(args.output),"w")
    for gold, pred in zip(y_dev, y_dev_pred):
        OUT.write("{}\t{}\n".format(gold,pred))
    OUT.close()
    OUT = open("{}.test.out".format(args.output), "w")
    for gold, pred in zip(y_test, y_test_pred):
        OUT.write("{}\t{}\n".format(gold, pred))
    OUT.close()

#print("Test dummy:")
#evaluate(y_test, y_test_dummy)
#print("Dev dummy:")
#evaluate(y_dev, y_dev_dummy)

#print(confusion_matrix(y_test,y_test_pred))

#print("Majority baseline:", accuracy_score(y_test, majority_prediction))
#print("majority label:", majority_label)
#show_most_informative_features(vectorizer,clf=classifier)

