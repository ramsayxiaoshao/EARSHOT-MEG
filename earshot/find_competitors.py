"""
Find competitors with a specific phonological relationships in the EARSHOT
lexicon.
"""
import pickle
from typing import List, Literal


from .earshot_lexicon import DATA_DIR, read_pronunciations


def read_competitors(
        lexicon: Literal['earshot', 'burgundy', 'MALD-NEIGHBORS-1000'],
        words: List[str] = None,
):
    if lexicon == 'burgundy':
        lexicon = 'MALD-NEIGHBORS-1000'
    path = DATA_DIR / f'competitors-{lexicon}'
    with path.open('rb') as file:
        competitors = pickle.load(file)
    if words is None or len(competitors) == len(words):
        return competitors
    # Remove words that are not in the lexicon
    out = {}
    for word in words:
        out[word] = {key: sorted(set(comps).intersection(words)) for key, comps in competitors[word].items()}
    return out


def find_competitors(
        lexicon: Literal['earshot', 'MALD-NEIGHBORS-1000'] = 'MALD-NEIGHBORS-1000',
):
    import speech.lexicon
    from trftools.dictionaries._arpabet import VOWELS

    pronuniciations = read_pronunciations(lexicon)
    lexicon_obj = speech.lexicon.generate_lexicon(pronuniciations)

    categories = [*range(8), 'rhyme', 'cohort', 'v-cohort', 'rhyme', 'unrelated']
    out = {}
    for word in lexicon_obj.words:
        pronunciation = word.pronunciations[0]
        competitors = {key: [] for key in categories}
        # find vowel index
        for i_vowel, ph in enumerate(pronunciation):
            if ph in VOWELS:
                break
        else:
            raise RuntimeError(f"{word!r} has no vowel")

        for other in lexicon_obj.words:
            other_pronunciation = other.pronunciations[0]
            if other_pronunciation == pronunciation:
                continue
            max_n = min(len(other_pronunciation), len(pronunciation))
            n_onset = n_offset = 0
            while n_onset < max_n and other_pronunciation.phones[n_onset] == pronunciation.phones[n_onset]:
                n_onset += 1
            while n_offset < max_n and other_pronunciation.phones[-n_offset - 1] == pronunciation.phones[-n_offset - 1]:
                n_offset += 1

            # N shared onset phonemes
            competitors[min(n_onset, 7)].append(other.graphs)
            # "Cohort": share first two phonemes
            if n_onset >= 2:
                competitors['cohort'].append(other.graphs)
                # Vowel-cohort: same up to first vowel
                if n_onset > i_vowel:
                    competitors['v-cohort'].append(other.graphs)
            # Rhyme: mismatch only in first phoneme
            elif len(other_pronunciation) == len(pronunciation) and n_offset == len(pronunciation) - 1:
                competitors['rhyme'].append(other.graphs)
            # Unrelated
            elif n_onset == 0 and n_offset == 0:
                competitors['unrelated'].append(other.graphs)
        out[word.graphs] = competitors

    path = DATA_DIR / f'competitors-{lexicon}'
    with path.open('wb') as file:
        pickle.dump(out, file, pickle.HIGHEST_PROTOCOL)
