"""Fixed synthetic recipe identity for model/consumer tests without corpus reads."""
from subtlex_mixture import configuration


def recipe(alpha=.5):
    return configuration(alpha, {'processed_sha256': '1' * 64, 'raw_sha256': '2' * 64,
                                 'normalization': {'unicode': 'NFKC then casefold', 'token_pattern': r'\w+'}})
