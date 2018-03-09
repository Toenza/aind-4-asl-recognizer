import warnings
from asl_data import SinglesData
import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb()  # initializes the database
asl.df.head()  # displays the first five rows of the asl database, indexed by video and frame
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary


from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))

# test the code
test_features_tryit(asl)

print(asl.df.ix[98,1])

# def recognize(models: dict, test_set: SinglesData):
#     """ Recognize test word sequences from word models set
#
#    :param models: dict of trained models
#        {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
#    :param test_set: SinglesData object
#    :return: (list, list)  as probabilities, guesses
#        both lists are ordered by the test set word_id
#        probabilities is a list of dictionaries where each key a word and value is Log Liklihood
#            [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
#             {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
#             ]
#        guesses is a list of the best guess words ordered by the test set word_id
#            ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
#    """
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     probabilities = []
#     guesses = []
#     # TODO implement the recognizer
#     # return probabilities, guesses
#     raise NotImplementedError
