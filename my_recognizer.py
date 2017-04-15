import warnings
from asl_data import SinglesData
import logging

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    all_Xlengths = test_set.get_all_Xlengths()
    for word_id in sorted(all_Xlengths):
        testword = test_set.wordlist[word_id]
        X, lengths = all_Xlengths[word_id]

        scores = {}
        for word in models:
            model = models[word]
            try:
                logL = model.score(X, lengths)
                scores[word] = logL
            except Exception as e:
                logging.warning('failed to score on {} using model {}'.format(testword, word))
                logging.warning(e)
                scores[word] = -float('inf')

        if len(scores) > 0:
            bestguess = max(scores, key = scores.get)
            probabilities.append(scores)
            guesses.append(bestguess)
        else:
            logging.warning('no score on word {}'.format(testword))
        
    return probabilities, guesses
