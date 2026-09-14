from __future__ import annotations

from cognitive_impairments.chat import clean_chat_text, parse_chat_text


def test__parse_chat_text__multiline_participant_utterance() -> None:
    """
    Arrange: CHAT-запись содержит перенос реплики участника и временную метку в продолжении.
    Act: запись разбирается общим парсером.
    Assert: текст, метаданные и границы реплики сохранены полностью.
    """
    content = """@Media:\t123-1, audio
@ID:\teng|Pitt|PAR|73;20|female|ProbableAD|||Participant|||
*PAR:\tthe boy is taking
\tcookies from the jar . \x151000_2500\x15
%mor:\tdet|the n|boy
"""

    result = parse_chat_text(content)

    assert result.metadata.media_id == '123-1'
    assert result.metadata.age == 73
    assert result.metadata.sex == 'female'
    assert result.metadata.diagnosis == 'ProbableAD'
    assert len(result.utterances) == 1
    assert result.utterances[0].text == 'the boy is taking cookies from the jar'
    assert result.utterances[0].start_ms == 1000
    assert result.utterances[0].end_ms == 2500


def test__clean_chat_text__unintelligible_tokens_do_not_damage_words() -> None:
    """
    Arrange: реплика содержит CHAT-маркеры, непонятный токен и обычное слово с буквой x.
    Act: текст очищается от разметки CHAT.
    Assert: служебные токены удалены, а обычное слово сохранено.
    """
    text = 'six xxx &um fallin(g) [//] down . \x151_2\x15'

    result = clean_chat_text(text)

    assert result == 'six um fallin down'


def test__parse_chat_text__missing_participant_metadata() -> None:
    """
    Arrange: CHAT-запись не содержит строки ID участника.
    Act: запись разбирается общим парсером.
    Assert: отсутствующие значения представлены явно, без исключения парсера.
    """
    content = '@Media:\t123-1, audio\n*PAR:\thello . \x151_2\x15\n'

    result = parse_chat_text(content)

    assert result.metadata.media_id == '123-1'
    assert result.metadata.age is None
    assert result.metadata.sex is None
    assert result.metadata.diagnosis is None
