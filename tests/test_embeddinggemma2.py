"""Whole-message embedding, cache resumption and cosine/visibility contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_ranker_features import cached_embeddings, cosine_scores, DIMENSIONS, MAX_TOKENS
from raw_datasets import memories_for_question


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.dtype = 'fake-float32'

    def lengths(self, texts, kind):
        return np.asarray([len(text.split()) + (0 if kind == 'content' else 8) for text in texts], dtype=np.int32)

    def encode(self, texts, kind):
        self.calls.append((list(texts), kind))
        values = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        values[:, 0] = 1
        return values


class WholeMessageTests(unittest.TestCase):
    def test_complete_long_message_is_one_embedding_and_resume_skips_it(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder()
            text = 'A complete source sentence. ' * 180 + 'Distinct original ending.'
            args = (Path(directory), 'memory', [text], ['original-message'], encoder, {'model': 'test'})
            vectors, lengths = cached_embeddings(*args)
            self.assertEqual(encoder.calls, [([text], 'memory')])
            self.assertGreater(int(lengths[0]), 256)
            self.assertEqual(vectors.shape, (1, DIMENSIONS))
            vectors._mmap.close()
            vectors, _ = cached_embeddings(*args)
            self.assertEqual(len(encoder.calls), 1)
            vectors._mmap.close()

    def test_oversized_message_is_rejected_before_any_text_is_encoded(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder()
            with self.assertRaisesRegex(ValueError, 'No splitting or truncation'):
                cached_embeddings(Path(directory), 'memory', ['word ' * MAX_TOKENS], ['long'], encoder, {})
            self.assertEqual(encoder.calls, [])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_1200_token_boundary_is_accepted_and_cache_records_limit(self):
        self.assertEqual(MAX_TOKENS, 1200)
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder()
            text = 'word ' * (MAX_TOKENS - 8)
            vectors, lengths = cached_embeddings(Path(directory), 'memory', [text], ['at-limit'], encoder, {})
            self.assertEqual(int(lengths[0]), 1200)
            self.assertEqual(len(encoder.calls), 1)
            metadata = json.loads((Path(directory) / 'memory_embedding_metadata.json').read_text())
            self.assertEqual(metadata['max_input_tokens'], 1200)
            vectors._mmap.close()

    def test_only_complete_visible_user_and_assistant_messages_are_scored(self):
        question = {'history_id': 'conversation', 'source_end_index': -1}
        contexts = {'conversation': [{'role': 'system', 'content': 'Private persona'},
                                     {'role': 'user', 'content': 'First sentence.\nSecond sentence.'},
                                     {'role': 'assistant', 'content': 'Complete answer.'},
                                     {'role': 'assistant'},
                                     {'role': 'user', 'content': 'Future message'}]}
        rows = memories_for_question(question, contexts)
        self.assertEqual([r['text'] for r in rows], ['First sentence.\nSecond sentence.', 'Complete answer.'])
        self.assertEqual([r['message_index'] for r in rows], [1, 2])

    def test_cosine_uses_normalized_vectors_and_preserves_negative_scores(self):
        vectors = np.asarray([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
        np.testing.assert_allclose(cosine_scores(vectors, np.asarray([1., 0.], dtype=np.float32)), [1., 0., -1.])


if __name__ == '__main__':
    unittest.main()
