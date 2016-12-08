import os
import unicodedata
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def transform_lang_data(lang_dir):
    data = []
    for file in os.listdir(lang_dir):
        text = [chr(l).lower() for l in
                unicodedata.normalize('NFKD', open(os.path.join(lang_dir, file)).read()).encode('ascii', 'ignore')
                if chr(l).isalpha()]
        data.append([text[i] + text[i + 1] for i in range(len(text) - 1)])
    return data


def lang_discovery(directory):
    ngrams_all = {}
    languages = []
    for lang in sorted(os.listdir(directory)):
        languages.append(lang)
        ngrams_all[lang] = transform_lang_data(os.path.join(directory, lang))

    ngr_counts = {}
    for lang, ngrams_all_samples in ngrams_all.items():
        for i, ngrams_sample in enumerate(ngrams_all_samples):
            c = Counter(ngrams_sample)



    vectors = {}
    for l in languages:
        vectors[l] = []
    sorted_ngrams = sorted(ngr_counts.keys())
    for ngr in sorted_ngrams:
        for l in languages:
            if l in ngr_counts[ngr].keys():
                vectors[l].append(ngr_counts[ngr][l])
            else:
                vectors[l].append(0)
                # print(ngr, ":", ngr_counts[ngr])

    for lang, v in vectors.items():
        print(lang, ":", v)
        # show_histogram(sorted_ngrams, v)

    for l in languages:
        arr = np.array(vectors[l])
        vectors[l] = arr * 100 / np.sum(arr)

    for lang, v in vectors.items():
        print(lang, ":", v)
        show_histogram(sorted_ngrams, v)


def show_histogram(sorted_ngrams, v):
    X = np.arange(len(v))
    plt.bar(X, v, align='center')
    plt.xticks(X, [ngr if i % 20 == 0 else '' for i, ngr in enumerate(sorted_ngrams)], rotation='vertical')
    ymax = np.max(v) + 1
    ymax = 10
    plt.ylim(0, ymax)
    plt.show()


if __name__ == '__main__':
    lang_discovery('data')
