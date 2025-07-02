# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from earshot.train_earshot import Lexicon
from earshot.earshot_lexicon import read_words


def test_mald_lexicon():
    lexicon = Lexicon('MALD-1000-train')
    burgundy_words = read_words('burgundy')
    mald_train_words = [w for s, w in lexicon.train_items if s == 'MALD']
    assert len(mald_train_words) > len(burgundy_words)
    assert all(w in mald_train_words for w in burgundy_words)
    test_words = [w for s, w in lexicon.test_items]
    assert len(test_words) == len(set(test_words))
    assert len(test_words) == lexicon.n_words
