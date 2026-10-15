import unittest
import torch
from torch.nn import functional as F
from PIL import Image
from captioning.model import Captioner, CaptionConfig
from captioning.vision import FrozenEncoder, preprocess_image


class CaptionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = Captioner(CaptionConfig(vocab_size=12, n_layer=2, n_embd=16,
            n_head=2, pmem_size=8, grid_size=2, feature_dim=3, max_caption_tokens=8,
            pad_id=0, bos_id=1, eos_id=2)).eval()
        self.features = torch.randn(2, 3, 2, 2)
        self.tokens = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])

    def test_causality_and_truncation(self):
        for flash in (True, False):
            for block in self.model.core.mosaic.h:
                block.attn.flash = flash
                if not flash:
                    block.attn.attn_dropout = torch.nn.Dropout(0)
            full, _ = self.model(self.features, self.tokens)
            changed = self.tokens.clone()
            changed[:, 2:] = 10
            perturbed, _ = self.model(self.features, changed)
            torch.testing.assert_close(full[:, :2], perturbed[:, :2])
            for length in range(1, 5):
                short, _ = self.model(self.features, self.tokens[:, :length])
                torch.testing.assert_close(full[:, :length], short, atol=1e-6, rtol=1e-5)

    def test_loss_and_gradients(self):
        targets = torch.tensor([[4, 5, 2, 0], [7, 2, 0, 0]])
        logits, loss = self.model(self.features, self.tokens, targets)
        valid = targets != 0
        torch.testing.assert_close(loss, F.cross_entropy(logits[valid], targets[valid]))
        loss.backward()
        self.assertGreater(self.model.adapter.weight.grad.abs().sum().item(), 0)
        self.assertIs(self.model.core.lm_head.weight, self.model.core.mosaic.wte.weight)
        altered = self.tokens.clone()
        altered[0, 3] = 11
        altered[1, 2:] = 11
        _, other_loss = self.model(self.features, altered, targets)
        torch.testing.assert_close(loss, other_loss)

    def test_example_independence_and_generation(self):
        batch, _ = self.model(self.features, self.tokens)
        self.model(self.features.flip(0), self.tokens.flip(0))
        single, _ = self.model(self.features[:1], self.tokens[:1])
        torch.testing.assert_close(batch[:1], single, atol=1e-6, rtol=1e-5)
        self.model.train()
        output = self.model.generate(self.features, max_new_tokens=4)
        self.assertTrue(self.model.training)
        self.assertLessEqual(output.shape[1], 5)
        self.assertTrue((output[:, 0] == 1).all())

    def test_preprocess_and_frozen_encoder(self):
        image = Image.new('RGB', (320, 100), color=(255, 0, 0))
        inputs = preprocess_image(image).unsqueeze(0)
        self.assertEqual(tuple(inputs.shape), (1, 3, 224, 224))
        encoder = FrozenEncoder(weights=None)  # No network required for this test.
        before = encoder.layers[1].running_mean.clone()
        encoder.train()
        features = encoder(inputs)
        self.assertFalse(encoder.training)
        self.assertFalse(features.requires_grad)
        self.assertEqual(tuple(features.shape), (1, 128, 14, 14))
        torch.testing.assert_close(before, encoder.layers[1].running_mean)

    def test_default_configuration_backward(self):
        model = Captioner(CaptionConfig(vocab_size=12, pad_id=0, bos_id=1, eos_id=2))
        _, loss = model(torch.randn(1, 128, 14, 14), self.tokens[:1],
                        torch.tensor([[4, 5, 6, 2]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
