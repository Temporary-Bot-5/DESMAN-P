"""Quote extraction and matching for document review."""

import pytest

from desman.quotes import missing, nearest, normalize, quotes


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


def test_replies_quoted_without_the_authors_words_between_them() -> None:
    # Found live (L.2, Chekhov): models quote dialogue the usual way, leaving out "-- he says --".
    source = [
        "Хлеб испекся.\n\xa0\xa0 -- Парня ненадежного послали! -- вздыхает хозяин, покачивая "
        "головой. -- Неповоротливый, как баба, должно, ветра испугался и не едет. Ишь ведь какой "
        "ветер!\n\xa0\xa0 -- Бе-ба! -- кричит в сенях дурачок. -- Ме-ма!"
    ]
    joined = (
        "Парня ненадежного послали! Неповоротливый, как баба, должно, ветра испугался и не едет."
    )
    assert missing([joined, "Бе-ба! Ме-ма! Хлеб испекся."], source) == [
        "Бе-ба! Ме-ма! Хлеб испекся."
    ]
    assert missing([joined], [source[0].replace("покачивая", "покачивая " + "очень " * 40)]) == [
        joined
    ]
    changed = "Парня ненадежного послали! Неповоротливый, как баба, ветра испугался."
    assert missing([changed], source) == [changed]


@pytest.mark.parametrize(
    "text",
    [
        'A 12" board goes next to a 14" board.',
        '{"description": "a lengthy JSON string", "other": 1}',
        '`print( "a lengthy code string" )`',
        '"a quote with\na line break"',
        '"a quote with {braces}"',
        '"a quote with a `backtick"',
        'word"no opening boundary"',
        '"no closing boundary"word',
        '" leading whitespace"',
    ],
)
def test_ascii_non_quotes_are_ignored(text: str) -> None:
    assert quotes(text) == []


def test_ascii_prose_quote_with_boundaries() -> None:
    assert quotes('He said ("a real quoted phrase"), then left.') == ["a real quoted phrase"]


def test_nearest_shows_what_the_text_really_says() -> None:
    # Found live (L.2): models kept writing "по нему" for Chekhov's "по нем" five times running.
    text = "Мутные валы со злобой хлещут по нем и тотчас же отскакивают назад, точно им гадко."
    quote = "Мутные валы со злобой хлещут по нему и тотчас же отскакивают назад"
    assert missing([quote], [text]) == [quote]
    found = nearest(quote, [text])
    assert found is not None and found.startswith("Мутные валы со злобой хлещут по нем и")
    assert nearest("Совсем другие слова здесь стоят", [text]) is None
