import pytest

from silksight.wikipedia import article_title, clean_extract, parse_intros


def test_article_title_decodes_language_and_title():
    url = "https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D0%B3%D0%B8%D1%81%D1%82%D0%B0%D0%BD_(X)"
    assert article_title(url) == ("ru", "Регистан (X)")
    with pytest.raises(ValueError):
        article_title("https://example.org/wiki/Registan")


def test_clean_extract_removes_empty_brackets_but_keeps_content():
    text = "Registan ( ) is a square (1420).\n\n\nIt has  three madrasas ; built later."

    assert clean_extract(text) == (
        "Registan is a square (1420).\n\nIt has three madrasas; built later."
    )


def test_parse_intros_maps_redirected_titles_back_to_the_requested_one():
    data = {
        "query": {
            "normalized": [{"from": "Gur-e_Amir", "to": "Gur-e Amir"}],
            "redirects": [{"from": "Gur-e Amir", "to": "Gur-e-Amir"}],
            "pages": [
                {
                    "title": "Gur-e-Amir",
                    "fullurl": "https://en.wikipedia.org/wiki/Gur-e-Amir",
                    "lastrevid": 42,
                    "extract": "Gur-e-Amir is a mausoleum.",
                },
                {"title": "Nothing", "missing": True},
            ],
        }
    }

    intros = parse_intros("en", data)

    assert intros["Gur-e_Amir"].revision_id == 42
    assert intros["Gur-e_Amir"].to_dict()["license"] == "CC BY-SA 4.0"
    assert "Nothing" not in intros
