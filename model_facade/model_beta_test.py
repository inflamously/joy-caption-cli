from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from model_facade import model_beta


class TestModelBeta:
    """Unit tests for model_beta module"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image at standard vision model input size"""
        # Create image at common vision model size to avoid resizing issues
        # LLaVA models typically use 336x336 or 384x384
        return Image.new('RGB', (384, 384), color=(255, 0, 0))

    @pytest.fixture
    def sample_images(self, sample_image):
        """Create a list of sample images"""
        return [sample_image]

    @pytest.fixture
    def mock_processor(self):
        def mock_decode(token, skip_special_tokens):
            if token[0] == 0:
                return "we"
            elif token[0] == 1:
                return " at"
            elif token[0] == 2:
                return " board"
            elif token[0] == 3:
                return " we"
            elif token[0] == 4:
                return ""
            elif token[0] == 5:
                return ","
            elif token[0] == 6:
                return " minim"
            elif token[0] == 7:
                return "al"
            elif token[0] == 8:
                return "istic"
            else:
                return None

        """Create a mock processor"""
        processor = Mock()
        processor.apply_chat_template.return_value = "mocked template"
        processor.batch_decode.return_value = ["A test caption"]
        processor.decode = mock_decode
        return processor

    def test_calculate_token_confidence(self):
        """Test token confidence calculation"""
        logits = torch.randn(5, 100)
        token_ids = torch.tensor([10, 20, 30, 40, 50])

        token_probs = model_beta.calculate_token_confidence(logits, token_ids)

        assert len(token_probs) == 5
        assert all(0.0 <= prob <= 1.0 for prob in token_probs)

    def test_calculate_entity_scores(self, mock_processor):
        """Test entity score calculation"""
        tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Realistic token IDs
        token_probs = [0.95, 0.23, 0.67, 0.89, 0.12, 0.54, 0.63, 0.31, 0.43, 0.56, 0.97]  # Mixed confidence

        entities = model_beta.calculate_entity_scores(tokens, token_probs, mock_processor)

        assert isinstance(entities, list)
        # check if "text" and "confidence" exists
        assert all('text' in ent and 'confidence' in ent for ent in entities)
        # check if "confidence" is between 0.0 and 1.0
        assert all(0.0 <= ent['confidence'] <= 1.0 for ent in entities)
        assert entities == [
            {'text': 'we', 'confidence': 0.95},
            {'text': 'board', 'confidence': 0.67},
            {'text': 'minimalistic', 'confidence': 0.4566666666666667}
        ]

    def test_inference_basic(self, sample_images):
        """Test basic inference functionality"""

        from model_facade.model_beta import load_llava
        processor, model = load_llava("fancyfeast/llama-joycaption-beta-one-hf-llava")

        results = model_beta.inference(
            processor=processor,
            model=model,
            images=sample_images,
            original_prompt="Test prompt",
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=50,
            show_prompt=False,
            batch_size=1,
            return_confidence_scores=False
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert 'joycaption' in results[0]
        assert 'prompt' in results[0]
        assert 'image' in results[0]
        assert "red" in results[0]["joycaption"]

    def test_inference_with_confidence_scores(self, sample_images):
        """Test inference with confidence scores enabled"""

        from model_facade.model_beta import load_llava
        processor, model = load_llava("fancyfeast/llama-joycaption-beta-one-hf-llava")

        results = model_beta.inference(
            processor=processor,
            model=model,
            images=sample_images,
            original_prompt="Test prompt",
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=50,
            show_prompt=False,
            batch_size=1,
            confidence_threshold=0.75,
            return_confidence_scores=True
        )

        assert isinstance(results, list)
        assert len(results) > 0
        result = results[0]
        assert 'overall_confidence' in result
        assert 'entities' in result
        assert 'filtered_caption' in result
        assert isinstance(result['overall_confidence'], float)
        assert isinstance(result['entities'], list)
        assert result["filtered_caption"] == "solid red background no objects color"

    def test_inference_with_none_image(self):
        """Test that inference raises error with None image"""

        from model_facade.model_beta import load_llava
        processor, model = load_llava("fancyfeast/llama-joycaption-beta-one-hf-llava")

        with pytest.raises(ValueError, match="One \\(or more\\) images was None"):
            model_beta.inference(
                processor=processor,
                model=model,
                images=[None],
                original_prompt="Test prompt",
                temperature=0.0,
                top_p=0.9,
                max_new_tokens=50,
                show_prompt=False,
                batch_size=1
            )

    def test_inference_empty_images(self):
        """Test inference with empty image list"""

        from model_facade.model_beta import load_llava
        processor, model = load_llava("fancyfeast/llama-joycaption-beta-one-hf-llava")

        results = model_beta.inference(
            processor=processor,
            model=model,
            images=[],
            original_prompt="Test prompt",
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=50,
            show_prompt=False,
            batch_size=1
        )

        assert isinstance(results, list)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
