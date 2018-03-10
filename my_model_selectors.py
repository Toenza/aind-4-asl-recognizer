import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        splits = 3

        split_method = KFold(random_state=self.random_state, n_splits=splits)
        best_score = float("-inf")
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                log_likelihood = model.score(self.X, self.lengths)
                d = len(self.X[0])
                # I have zero idea why this is like that and why we need to figure this out on our own
                # but according to https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
                # the number of parameters is n^2 + 2*d*n - 1 (where d is the #features and n #states in the HMM)
                n_params = n**2 + 2*d*n - 1
                N = len(self.X)

                # BIC = -2 * logL + p * logN     (source: http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf)
                BIC = -2 * log_likelihood + n_params * np.log(N)
                if BIC < best_score:
                    best_score = BIC
                    best_model = model
            except Exception as e:
                continue

        return best_model if best_model is not None else self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = {}
        likelihoods = {}

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            n_models = {}
            n_likelihoods = {}

            for word in self.words.keys():
                X, lengths = self.hwords[word]
                try:
                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_set, lengths_train)
                    log_likelihood = model.score(X, lengths)
                    n_models[word] = model
                    n_likelihoods[word] = log_likelihood
                except Exception as e:
                    continue

            models[n_components] = n_models
            likelihoods[n_components] = n_likelihoods

        best_model = None
        best_score = float("-inf")

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = models[n_components]
            likelihood = likelihoods[n_components]

            if self.this_word not in likelihood:
                continue

            other_words = [likelihood[word] for word in likelihood.keys() if word != self.this_word]
            DIC = likelihood[self.this_word] - np.mean(other_words)

            if DIC > best_score:
                best_model = model[self.this_word]
                best_score = DIC

        return best_model if best_model is not None else self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        splits = 3

        split_method = KFold(random_state=self.random_state, n_splits=splits)
        best_score = float("-inf")
        best_model = None

        if len(self.sequences) >= splits:
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                scores = []
                model = None
                log_likelihood = None
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_set, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    test_set, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_set, lengths_train)
                        log_likelihood = model.score(test_set, lengths_test)
                        scores.append(log_likelihood)
                    except Exception as e:
                        break

                average_score = np.average(scores) if len(scores) > 0 else float("-inf")
                if average_score > best_score:
                    best_score, best_model = average_score, model

        return best_model if best_model is not None else self.base_model(self.n_constant)
