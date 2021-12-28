

from os import walk
from math import log
import itertools
import builtins


def read_data():
    data = []
    for i in range(10):
        path = './pu1/part%d/' % (i + 1)
        files = next(walk(path))[2]

        t = []
        for file in files:
            t.append(process_file(path + file))
        data.append(t)
    return data


def process_file(name):
    with open(name, 'r') as f:
        a = "".join(f.readlines()).split()
        a = a[1:]
        a = list(map(int, a))
        return (a, int('spmsg' in name))


def F1(out, cond):

    tp = fp = fn = 0
    for i in range(len(out)):
        o = out[i]
        c = cond[i]

        if o == 1 and c == 1:
            tp += 1

        if o == 1 and c == 0:
            fp += 1

        if o == 0 and c == 1:
            fn += 1

    # precision = float(tp) / (tp + fp)
    # recall = float(tp) / (tp + fn)

    # f1 = 2 * (precision) * (recall) / (precision + recall)

    f1 = 2.0 * tp / (2.0 * tp + fp + fn)

    return f1


def naive_train(data):
    classes = [0, 0]
    cnt_words = [0, 0]
    freq = {}
    words = set()
    for (features, label) in data:
        classes[label] += 1

        for feature in features:
            freq[label, feature] = freq.get((label, feature), 0) + 1
            cnt_words[label] += 1
            words.add(feature)

    for i in range(2):
        classes[i] = classes[i] / float(len(data))

    cnt_dict = len(words)

    return classes, freq, cnt_words, cnt_dict


def naive_classify(classifier, features):
    classes, freq, cnt_words, cnt_dict = classifier
    prob = [0, 0]
    for c in range(2):
        a = log(classes[c])
        for feat in features:
            t = (freq.get((c, feat), 0) + 1) / (cnt_dict + cnt_words[c])
            a += log(t)
        prob[c] = a
    return 0 if prob[0] > prob[1] else 1


def cross_validation(data):
    f_sum = 0.0
    for i in range(len(data)):
        t = data[0:i] + data[i + 1:]
        train = list(itertools.chain(*t))
        test = data[i]

        classifier = naive_train(train)

        cond = [sample[1] for sample in test]
        out = []

        for sample in test:
            cl = naive_classify(classifier, sample[0])
            out.append(cl)

        f1 = F1(out, cond)
        f_sum += f1
    f_mean = f_sum / len(data)
    return f_mean


def main():
    data = read_data()
    f_mean = cross_validation(data)
    print(f_mean)


if __name__ == '__main__':
    main()