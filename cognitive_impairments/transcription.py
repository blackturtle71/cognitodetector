from __future__ import annotations

import pathlib


def transcribe_audio(
    audio_path: pathlib.Path,
    output_path: pathlib.Path,
    model_name: str,
    language: str | None,
) -> None:
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite transcript: {output_path}')

    import whisper

    model = whisper.load_model(model_name)
    options = {'language': language} if language is not None else {}
    result = model.transcribe(str(audio_path), **options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result['text'].strip() + '\n', encoding='utf-8')
