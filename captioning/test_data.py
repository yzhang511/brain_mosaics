import random
import unittest

from captioning.data import BOS_ID, EOS_ID, PAD_ID, caption_batch, decode, encode, select_captions


class DataTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        ids = encode('A cat sits.')
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(ids[-1], EOS_ID)
        self.assertEqual(decode(ids), 'A cat sits.')

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            encode('one two three', max_tokens=3)

    def test_special_ids_never_collide_with_real_tokens(self):
        # encode_ordinary never emits ids >= eot, so BOS/PAD sentinels (just above it) are safe.
        for text in ('a photo of a cat', 'a man riding a bicycle down the street'):
            self.assertNotIn(BOS_ID, encode(text)[1:-1])
            self.assertNotIn(PAD_ID, encode(text)[1:-1])
            self.assertNotIn(EOS_ID, encode(text)[1:-1])

    def test_caption_batch_pads_with_pad_id_not_zero(self):
        records = [{'caption': 'a cat'}, {'caption': 'a much longer caption about a dog running'}]
        inputs, targets = caption_batch(records)
        self.assertEqual(inputs[0, 0].item(), BOS_ID)
        self.assertTrue((inputs[0, len(encode(records[0]['caption'])) - 1:] == PAD_ID).all())
        self.assertTrue((targets[0, len(encode(records[0]['caption'])) - 1:] == PAD_ID).all())

    def test_select_captions_is_deterministic_given_rng(self):
        records = [{'all_captions': ['a', 'b', 'c']}, {'all_captions': ['x', 'y']}]
        first = select_captions(records, random.Random(0))
        second = select_captions(records, random.Random(0))
        self.assertEqual(first, second)
        self.assertTrue(all(c in r['all_captions'] for c, r in zip(first, records)))


if __name__ == '__main__':
    unittest.main()
