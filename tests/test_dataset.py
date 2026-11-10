import io

import pytest
from PIL import Image

from silksight.commons import CommonsImage, plain_text
from silksight.dataset import (
    DatasetError,
    assign_splits,
    is_allowed_license,
    normalize_image,
    select_diverse,
)


def _image(title: str, author: str) -> CommonsImage:
    return CommonsImage(
        title,
        "page",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/x.jpg/960px-x.jpg",
        1,
        1,
        author,
        "",
        "",
    )


@pytest.mark.parametrize(
    ("license_name", "allowed"),
    [
        ("CC BY-SA 4.0", True),
        ("CC BY 2.0", True),
        ("CC0", True),
        ("Public domain", True),
        ("CC BY-NC-SA 2.0", False),
        ("CC BY-ND 3.0", False),
        ("All rights reserved", False),
        ("GFDL", False),
    ],
)
def test_license_filter(license_name, allowed):
    assert is_allowed_license(license_name) is allowed


def test_author_html_becomes_plain_text():
    assert plain_text('<a href="//x">Jane&nbsp;Doe</a> (talk)') == "Jane Doe (talk)"


def test_select_diverse_round_robins_authors():
    images = [_image(f"a{i}", "A") for i in range(10)] + [_image("b0", "B"), _image("c0", "C")]
    selected = select_diverse(images, 4)
    assert {image.author for image in selected} == {"A", "B", "C"}


def test_splits_never_share_an_author_and_keep_train_majority():
    authors = ["big"] * 20 + [f"solo{i}" for i in range(10)] + ["pair"] * 2
    splits = assign_splits(authors, "registan", (("test", 5), ("val", 3)))

    by_author: dict[str, set[str]] = {}
    for author, split in zip(authors, splits):
        by_author.setdefault(author, set()).add(split)
    assert all(len(values) == 1 for values in by_author.values())
    assert splits.count("test") >= 5
    assert splits.count("val") >= 3
    assert splits.count("train") >= len(authors) / 2
    assert by_author["big"] == {"train"}


def test_one_large_small_group_does_not_starve_validation():
    authors = ["a", "b", "c", "d"] + ["e"] * 5 + ["f"] * 25
    splits = assign_splits(authors, "gur-e-amir", (("test", 5), ("val", 3)))
    assert splits.count("val") >= 2
    assert splits.count("test") >= 5


def test_single_author_class_gets_no_held_out_split():
    assert set(assign_splits(["only"] * 10, "x", (("test", 5),))) == {"train"}


def test_normalize_image_applies_orientation_and_drops_metadata():
    source = Image.new("RGB", (400, 300), "red")
    exif = source.getexif()
    exif[0x0112] = 6  # rotate 90 degrees when displayed
    buffer = io.BytesIO()
    source.save(buffer, "JPEG", exif=exif)

    with Image.open(io.BytesIO(normalize_image(buffer.getvalue()))) as result:
        assert result.size == (300, 400)
        assert not result.getexif()


def test_tiny_or_broken_images_are_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buffer, "PNG")
    with pytest.raises(DatasetError, match="too small"):
        normalize_image(buffer.getvalue())
    with pytest.raises(DatasetError, match="not a readable image"):
        normalize_image(b"not an image")
