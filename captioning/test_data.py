import unittest
from captioning.data import build_vocabulary, caption_batch, encode, decode


class DataTests(unittest.TestCase):
    def test_train_vocabulary_and_alignment(self):
        vocab = build_vocabulary(['A cat sits.', 'A dog sits.'])
        self.assertEqual(vocab, build_vocabulary(['A dog sits.', 'A cat sits.']))
        self.assertEqual(encode('unseen', vocab), [1, 3, 2])
        x, y = caption_batch([{'caption': 'A cat sits.'}, {'caption': 'dog'}], vocab)
        self.assertEqual(x[0, 0].item(), 1)
        self.assertEqual(y[0, -1].item(), 2)
        self.assertEqual(y[1].tolist(), [vocab.index('dog'), 2, 0, 0, 0])
        self.assertEqual(decode(encode('A cat sits.', vocab), vocab), 'a cat sits .')

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            encode('one two three', build_vocabulary(['one two three']), max_tokens=3)


if __name__ == '__main__':
    unittest.main()
