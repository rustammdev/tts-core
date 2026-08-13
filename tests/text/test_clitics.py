from uztts_text.clitics import join_clitics


def test_joins_split_clitics() -> None:
    assert join_clitics("qildik da yaxshi boʻldi ku") == "qildikda yaxshi boʻldiku"


def test_counter_ta_joins() -> None:
    assert join_clitics("oʻn besh ta odam") == "oʻn beshta odam"


def test_question_mi_joins() -> None:
    assert join_clitics("keldingiz mi") == "keldingizmi"


def test_already_joined_untouched() -> None:
    assert join_clitics("qildikda beshta keldimi") == "qildikda beshta keldimi"


def test_leading_clitic_token_kept() -> None:
    assert join_clitics("da boshlandi") == "da boshlandi"


def test_pronoun_u_not_joined() -> None:
    assert join_clitics("men u kitobni oldim") == "men u kitobni oldim"


def test_empty_text() -> None:
    assert join_clitics("") == ""
