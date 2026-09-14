from __future__ import annotations

import dataclasses
import pathlib
import re


TIMECODE_PATTERN = re.compile(r'\x15(?P<start>\d+)_(?P<end>\d+)\x15')
BRACKET_ANNOTATION_PATTERN = re.compile(r'\[[^\]]*\]')
PARENTHESIZED_SUFFIX_PATTERN = re.compile(r'\((\w+)\)')
PARENTHESIZED_ANNOTATION_PATTERN = re.compile(r'\([^)]*\)')
CHAT_MARKER_PATTERN = re.compile(r'[&+\-^]+(?=\w)')
UNINTELLIGIBLE_PATTERN = re.compile(r'\bx+\b', flags=re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class Utterance:
    text: str
    start_ms: int
    end_ms: int


@dataclasses.dataclass(frozen=True)
class ParticipantMetadata:
    media_id: str | None
    age: int | None
    sex: str | None
    diagnosis: str | None


@dataclasses.dataclass(frozen=True)
class ChatRecord:
    metadata: ParticipantMetadata
    utterances: tuple[Utterance, ...]


def clean_chat_text(text: str) -> str:
    text = TIMECODE_PATTERN.sub(' ', text)
    text = BRACKET_ANNOTATION_PATTERN.sub(' ', text)
    text = PARENTHESIZED_SUFFIX_PATTERN.sub('', text)
    text = PARENTHESIZED_ANNOTATION_PATTERN.sub(' ', text)
    text = CHAT_MARKER_PATTERN.sub('', text)
    text = UNINTELLIGIBLE_PATTERN.sub(' ', text)
    text = re.sub(r'[<>/.,!?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _parse_participant_id(value: str | None) -> tuple[int | None, str | None, str | None]:
    if value is None:
        return None, None, None

    fields = value.split('|')
    if len(fields) < 6:
        return None, None, None

    age_text = fields[3].split(';', maxsplit=1)[0].strip()
    age = int(age_text) if age_text.isdigit() else None
    sex = fields[4].strip().lower() or None
    diagnosis = fields[5].strip() or None
    return age, sex, diagnosis


def _parse_utterance(raw_text: str) -> Utterance | None:
    timecode = TIMECODE_PATTERN.search(raw_text)
    if timecode is None:
        return None

    text = clean_chat_text(raw_text)
    if not text:
        return None

    start_ms = int(timecode.group('start'))
    end_ms = int(timecode.group('end'))
    if end_ms <= start_ms:
        return None
    return Utterance(
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def parse_chat_text(content: str) -> ChatRecord:
    metadata: dict[str, str] = {}
    participant_id: str | None = None
    utterances: list[Utterance] = []
    current_participant_lines: list[str] = []

    def flush_participant_utterance() -> None:
        if not current_participant_lines:
            return
        utterance = _parse_utterance(' '.join(current_participant_lines))
        if utterance is not None:
            utterances.append(utterance)
        current_participant_lines.clear()

    for line in content.splitlines():
        if line.startswith('*'):
            flush_participant_utterance()
            tier, _, value = line.partition(':')
            if tier == '*PAR':
                current_participant_lines.append(value.strip())
            continue

        if line.startswith('\t') and current_participant_lines:
            current_participant_lines.append(line.strip())
            continue

        if line.startswith('@'):
            flush_participant_utterance()
            key, separator, value = line[1:].partition(':')
            if not separator:
                continue
            value = value.strip()
            if key == 'ID':
                fields = value.split('|')
                if len(fields) > 2 and fields[2] == 'PAR':
                    participant_id = value
            else:
                metadata[key] = value
            continue

        if line.startswith('%'):
            flush_participant_utterance()

    flush_participant_utterance()
    age, sex, diagnosis = _parse_participant_id(participant_id)
    media = metadata.get('Media')
    media_id = media.split(',', maxsplit=1)[0].strip() if media else None
    return ChatRecord(
        metadata=ParticipantMetadata(
            media_id=media_id,
            age=age,
            sex=sex,
            diagnosis=diagnosis,
        ),
        utterances=tuple(utterances),
    )


def parse_chat_file(path: pathlib.Path) -> ChatRecord:
    content = path.read_text(encoding='utf-8', errors='ignore')
    return parse_chat_text(content)
