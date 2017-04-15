from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import *
from asl_data import AslDb
import timeit
from lm import create2gram

import logging
logging.basicConfig(filename='batch.log', filemode='w', level=logging.DEBUG)

def select_model_for_word_with_features(word, features, model_selector):
    training = asl.build_training(features)
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()

    start = timeit.default_timer()
    model = model_selector(
            sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state=14).select()
    end = timeit.default_timer() - start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(
            word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict


asl = AslDb() # initializes the database
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
# c.log
# features_custom = ['norm-delta-rx', 'norm-delta-ry', 'norm-delta-lx', 'norm-delta-ly']
# d.log
features_custom1 = ['norm-grnd-rx', 'norm-grnd-ry', 'norm-grnd-lx', 'norm-grnd-ly']
# g.log
features_custom2 = ['delta-grnd-rx', 'delta-grnd-ry', 'delta-grnd-lx', 'delta-grnd-ly']
features_custom3 = ['delta-polar-rr', 'delta-polar-rtheta', 'delta-polar-lr', 'delta-polar-ltheta']

# features_ground
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']


# features_norm
features = ['right-x','right-y','left-x','left-y']
for i in range(4):
    tmpfeature = features[i]
    tmpmean = asl.df['speaker'].map(df_means[tmpfeature])
    tmpstd = asl.df['speaker'].map(df_std[tmpfeature])
    asl.df[features_norm[i]] = (asl.df[tmpfeature] - tmpmean) / tmpstd

# features_polar
rx = asl.df['right-x'] - asl.df['nose-x']
ry = asl.df['right-y'] - asl.df['nose-y']
lx = asl.df['left-x'] - asl.df['nose-x']
ly = asl.df['left-y'] - asl.df['nose-y']
asl.df['polar-rr'] = np.sqrt(rx**2 + ry**2)
asl.df['polar-rtheta'] = np.arctan2(rx, ry)
asl.df['polar-lr'] = np.sqrt(lx**2 + ly**2)
asl.df['polar-ltheta'] = np.arctan2(lx, ly)

# features_delta
features = ['right-x', 'right-y', 'left-x', 'left-y']
for i in range(4):
    tmpfeature = features[i]
    asl.df[features_delta[i]] = asl.df[tmpfeature].diff().fillna(0.0)

# features_custom1: normallized grnd
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()
features = features_ground
for i in range(4):
    tmpfeature = features[i]
    tmpmean = asl.df['speaker'].map(df_means[tmpfeature])
    tmpstd = asl.df['speaker'].map(df_std[tmpfeature])
    asl.df[features_custom1[i]] = (asl.df[tmpfeature] - tmpmean) / tmpstd

# features_custom2: grnd delta
features = features_ground
for i in range(4):
    tmpfeature = features[i]
    asl.df[features_custom2[i]] = asl.df[tmpfeature].diff().fillna(0.0)

# features_custom3: polar delta
features = features_polar
for i in range(4):
    tmpfeature = features[i]
    asl.df[features_custom3[i]] = asl.df[tmpfeature].diff().fillna(0.0)

features_custom4 = features_ground + features_polar

if __name__ == '__main__':
    #select_model_for_word_with_features('FIND', features_polar, SelectorBIC)
    #sys.exit()
    # featureset = [features_norm, features_polar, features_ground, features_delta, features_custom]
    # selectorset = [SelectorCV, SelectorBIC, SelectorDIC]
    featureset = [features_polar, features_ground]
    selectorset = [SelectorBIC, SelectorDIC]
    start = timeit.default_timer()
    bigram = create2gram()
    lmscale = 150
    for features in featureset:
        for model_selector in selectorset:
            print('train_all_words using features {} and selector {}'.format(features, model_selector))
            models = train_all_words(features, model_selector)
            test_set = asl.build_test(features)
            probabilities, guesses = recognize(models, test_set)
            print('show_errors on features: {}, selector: {}'.format(features, model_selector))
            show_errors(guesses, test_set)
    end = timeit.default_timer() - start
    print('recognizer finished in {} seconds'.format(end))
