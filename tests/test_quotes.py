"""Quote extraction and matching for document review."""

import pytest

from desman.quotes import missing, normalize, quotes


def test_extracts_all_four_styles_in_order() -> None:
    assert quotes(
        '« a long quotation » „third quotation“ “another quotation” "last quotation" «short»'
    ) == [
        "a long quotation",
        "third quotation",
        "another quotation",
        "last quotation",
    ]


def test_normalizes_spelling_and_typography() -> None:
    assert normalize(" «ЁЖ —–‑- идёт\n к е́ли!» ") == "еж - идет к ели"


@pytest.mark.parametrize(
    ("quote", "sources", "found"),
    [
        ("а на деле выходим — волки", ["а на деле выходим -- волки"], True),
        ("жестки на ощупь", ['"жестка на ощупь"'], False),
        ("На берегу стоял… Вдали был лес", ["На берегу стоял дом.", "Вдали был лес."], True),
        ("На берегу стоял... Вдали был лес", ["На берегу стоял дом."], False),
        (
            "Томск — город удивительный, не сибирский, а европейский",
            ["В Томске грязно. А на деле выходим -- волки."],
            False,
        ),
    ],
)
def test_matches_verbatim_fragments(quote: str, sources: list[str], found: bool) -> None:
    assert missing([quote], sources) == ([] if found else [quote])


def test_a_title_in_another_language_is_not_a_quote_to_check() -> None:
    source = ["Из Сибири. Отчего у вас в Сибири так холодно? -- Богу так угодно!"]
    assert missing(["From Siberia, travel notes"], source) == []
    assert missing(["Отчего у вас в Сибири так жарко"], source) == [
        "Отчего у вас в Сибири так жарко"
    ]
