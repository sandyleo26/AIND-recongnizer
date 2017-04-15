import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import logging

def create_default_hmm(num_states, random_state):
    return GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
            random_state=random_state, verbose=False)

def create_left_right_hmm(num_states, random_state):
    '''
    create left to right hmm like in the lecture (state1 -> state2 -> state3...)
    Always start from state1
    '''
    transmat = np.zeros((num_states, num_states))
    for i in range(num_states):
        if i == num_states-1:
            transmat[i, i] = 1.0
        else:
            transmat[i, i] = 0.5
            transmat[i, i+1] = 0.5
    startprob = np.zeros(num_states)
    startprob[0] = 1.0
    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
            random_state=random_state, verbose=False,
            params='mct', init_params='cm')
    model.startprob_ = startprob
    model.transmat_ = transmat
    return model

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection based on BIC scores
        results = []
        model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            bic = float('inf')
            try:
                # model = create_default_hmm(num_states=num_states, random_state=self.random_state)
                model = create_left_right_hmm(num_states=num_states, random_state=self.random_state)
                model.fit(self.X, self.lengths)

                # 1. full covars, no fixed (transition matrix + start states + emission Gaussian + full covar)
                # num_params = num_states * (num_states - 1) + num_states - 1 + num_states * (
                #         len(self.X[0]) + 1) * len(self.X[0])
                # 2. first-order Markov chain, 'diag', fixed start1 (transition matrix + emission Gaussian)
                num_params = num_states - 1 + 2 * num_states * len(self.X[0])

                logL = model.score(self.X, self.lengths)
                bic = -2 * logL + num_params * math.log(len(self.X))
                logging.debug('{} with {} states bic {}, logL {}, num_params {}'.format(
                    self.this_word, num_states, bic, logL, num_params))
                results.append((bic, model))
            except Exception as e:
                logging.warning("failure on {} with {} states".format(self.this_word, num_states))
                logging.warning(e)

        if len(results) > 0:
            bic, model = min(results, key = lambda x: x[0])
            logging.info("model created for {} with {} states and score: {}".format(
                self.this_word, model.n_components, bic))
        else:
            logging.warning('SelectorBIC returns nothing for word {}'.format(self.this_word))
        return model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection based on DIC scores
        results = []
        model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            dic = -float('inf')
            try:
                # model = create_default_hmm(num_states=num_states, random_state=self.random_state)
                model = create_left_right_hmm(num_states=num_states, random_state=self.random_state)
                model.fit(self.X, self.lengths)

                logL = model.score(self.X, self.lengths)
                other_logLs = []
                for other_word in self.hwords:
                    if other_word == self.this_word:
                        continue
                    otherX, otherlengths = self.hwords[other_word]
                    other_logL = model.score(otherX, otherlengths)
                    other_logLs.append(other_logL)

                avg_other_logL = np.mean(other_logLs)
                dic = logL - avg_other_logL
                results.append((dic, model))
            except Exception as e:
                logging.warning("failure on {} with {} states".format(self.this_word, num_states))
                logging.warning(e)

        if len(results) > 0:
            dic, model = max(results, key = lambda x: x[0])
            logging.info("model created for {} with {} states and score: {}".format(
                self.this_word, model.n_components, dic))
        else:
            logging.warning('SelectorDIC returns nothing for word {}'.format(self.this_word))
        return model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection using CV
        split_method = KFold()
        results = []
        model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                logL = -float('inf')
                if len(self.sequences) < 3:
                    # model = create_default_hmm(num_states=num_states, random_state=self.random_state)
                    model = create_left_right_hmm(num_states=num_states, random_state=self.random_state)
                    model.fit(self.X, self.lengths)

                    logL = model.score(self.X, self.lengths)
                else:
                    scores = []
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # view indices of the folds
                        train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                        # model = create_default_hmm(num_states=num_states, random_state=self.random_state)
                        model = create_left_right_hmm(num_states=num_states, random_state=self.random_state)
                        model.fit(train_X, train_lengths)

                        tmplogL = model.score(test_X, test_lengths)
                        scores.append(tmplogL)
                    logL = np.mean(scores)
                    logging.info('average score on {} with {} states is {}'.format(
                        self.this_word, num_states, logL))
                results.append((logL, model))
            except Exception as e:
                logging.warning("failure on {} with {} states".format(self.this_word, num_states))
                logging.warning(e)

        if len(results) > 0:
            logL, model = max(results, key = lambda x: x[0])
            logging.info("model created for {} with {} states and score: {}".format(
               self.this_word, model.n_components, logL))
        else:
            logging.warning('SelectorCV returns nothing for word {}'.format(self.this_word))
        return model
