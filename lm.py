from copy import deepcopy
import logging

# the log probability is 10 based
def create2gram():
    filename = 'resource/lm/devel-lm-M2.sri.lm'
    f = open(filename)
    bigram_start = False
    bigram = {}
    for line in f:
        line = line.strip()
        # print(line)
        if line == '\\2-grams:':
            bigram_start = True
            continue
        elif bigram_start and not line:
            break

        if bigram_start:
            logStr, preceding, word = line.split()
            if preceding not in bigram:
                bigram[preceding] = {}

            bigram[preceding][word] = float(logStr)

    return bigram

# TODO
# http://www.foldl.me/2014/kneser-ney-smoothing/
# http://www.speech.sri.com/projects/srilm/manpages/ngram-format.5.html

def create2gram_knsmoothed():
    filename = 'resource/lm/devel-lm-M2.sri.lm'
    theta = 0.75
    f = open(filename)
    unigram_start = False
    bigram_start = False
    unigram = {}
    bigram = {}
    unigram_total = 0
    bigram_total = 0
    for line in f:
        line = line.strip()
        # print(line)
        if line.startswith('ngram 1'):
            unigram_total = int(line.split('=')[1])
        elif line.startswith('ngram 2'):
            bigram_total = int(line.split('=')[1])

        if line == '\\1-grams:':
            unigram_start = True
            continue
        elif unigram_start and not line:
            unigram_start = False
            continue

        if line == '\\2-grams:':
            bigram_start = True
            continue
        elif bigram_start and not line:
            break

        if unigram_start:
            words = line.split()
            if len(words) == 2 or words[0] == '-99':
                continue
            unigram[words[1]] = float(words[0]) + float(words[2])

        elif bigram_start:
            logStr, preceding, word = line.split()
            if preceding not in bigram:
                bigram[preceding] = {}

            bigram[preceding][word] = float(logStr)

    newbigram = deepcopy(bigram)
    # kn smoothing
    # for uni in newbigram:
    #     if uni not in newbigram:
    #         logging.debug('uni word {} not in bigram'.format(uni))
    #         newbigram[uni] = {}

    return newbigram


if __name__ == '__main__':
    print('Hello World')
    lm = create2gram()
    # bigram = create2gram_knsmoothed()
