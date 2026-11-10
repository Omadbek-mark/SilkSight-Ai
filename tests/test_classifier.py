import pytest

from silksight.classifier import class_weights


def test_class_weights_favour_small_classes_and_average_one():
    weights = class_weights(["a", "a", "a", "b"], ["a", "b"])

    assert weights[1] == pytest.approx(3 * weights[0])
    assert float(weights.mean()) == pytest.approx(1.0)


def test_class_weights_reject_class_without_training_images():
    with pytest.raises(ValueError, match="no training images for: b"):
        class_weights(["a"], ["a", "b"])
