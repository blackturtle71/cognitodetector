from __future__ import annotations

import argparse
import pathlib


DEFAULT_CONFIG_PATH = pathlib.Path('configs/pitt_multimodal.toml')
DEFAULT_DATA_ROOT = pathlib.Path('data/DementiaBank')


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m cognitive_impairments',
        description='Reproducible cognitive-impairment experiments from speech.',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    preprocess = commands.add_parser('preprocess', help='Build a canonical feature table.')
    preprocess.add_argument('corpus', choices=('pitt', 'adresso'))
    preprocess.add_argument('--data-root', type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    preprocess.add_argument('--output', type=pathlib.Path, required=True)
    preprocess.add_argument('--config', type=pathlib.Path, default=DEFAULT_CONFIG_PATH)
    preprocess.add_argument('--task', default='cookie')
    preprocess.add_argument('--split', choices=('train', 'test'), default='train')
    _add_device_argument(preprocess)

    train = commands.add_parser('train', help='Run the grouped dissertation experiment suite.')
    train.add_argument('--data', type=pathlib.Path, required=True)
    train.add_argument('--config', type=pathlib.Path, default=DEFAULT_CONFIG_PATH)
    train.add_argument('--output', type=pathlib.Path, required=True)
    train.add_argument('--seed', type=int, required=True)
    _add_device_argument(train)

    paper_run = commands.add_parser(
        'paper-run',
        help='Run or resume the prespecified paper evaluation protocol.',
    )
    paper_run.add_argument('--data', type=pathlib.Path, required=True)
    paper_run.add_argument('--config', type=pathlib.Path, required=True)
    paper_run.add_argument('--output', type=pathlib.Path, required=True)
    paper_run.add_argument('--report-output', type=pathlib.Path, required=True)
    paper_run.add_argument('--resume', action='store_true')
    _add_device_argument(paper_run)

    statistics = commands.add_parser('stats', help='Save dataset statistics as JSON.')
    statistics.add_argument('--data', type=pathlib.Path, required=True)
    statistics.add_argument('--output', type=pathlib.Path, required=True)

    transcribe = commands.add_parser('transcribe', help='Transcribe one recording with Whisper.')
    transcribe.add_argument('--audio', type=pathlib.Path, required=True)
    transcribe.add_argument('--output', type=pathlib.Path, required=True)
    transcribe.add_argument('--model', default='base')
    transcribe.add_argument('--language')

    explain = commands.add_parser('explain', help='Explain held-out fold predictions with SHAP.')
    explain.add_argument('--run', type=pathlib.Path, required=True)
    explain.add_argument('--data', type=pathlib.Path, required=True)
    _add_device_argument(explain)

    predict = commands.add_parser('predict', help='Predict with the multimodal fold ensemble.')
    prediction_modes = predict.add_subparsers(dest='prediction_mode', required=True)
    predict_table_parser = prediction_modes.add_parser('table')
    predict_table_parser.add_argument('--run', type=pathlib.Path, required=True)
    predict_table_parser.add_argument('--data', type=pathlib.Path, required=True)
    predict_table_parser.add_argument('--output', type=pathlib.Path, required=True)
    predict_table_parser.add_argument('--adresso-format', action='store_true')
    _add_device_argument(predict_table_parser)

    predict_recording_parser = prediction_modes.add_parser('recording')
    predict_recording_parser.add_argument('--run', type=pathlib.Path, required=True)
    predict_recording_parser.add_argument('--audio', type=pathlib.Path, required=True)
    predict_recording_parser.add_argument('--transcript', type=pathlib.Path, required=True)
    predict_recording_parser.add_argument('--age', type=int, required=True)
    predict_recording_parser.add_argument('--sex', choices=('female', 'male'), required=True)
    predict_recording_parser.add_argument('--output', type=pathlib.Path, required=True)
    _add_device_argument(predict_recording_parser)
    return parser


def _run_preprocess(arguments: argparse.Namespace) -> None:
    from cognitive_impairments.config import load_experiment_config
    from cognitive_impairments.features import TextEmbedder
    from cognitive_impairments.preprocessing import prepare_adresso, prepare_pitt, resolve_device

    config = load_experiment_config(arguments.config)
    device = resolve_device(arguments.device)
    embedder = TextEmbedder(
        model_name=config.features.embedder,
        revision=config.features.revision,
        pooling=config.features.pooling,
        normalize_embedding=config.features.normalize,
        max_length=config.features.max_length,
        device=device,
    )
    if arguments.corpus == 'pitt':
        prepare_pitt(
            arguments.data_root,
            arguments.output,
            embedder,
            arguments.task,
            config.features.acoustic_aggregation,
        )
    else:
        prepare_adresso(
            arguments.data_root,
            arguments.output,
            embedder,
            arguments.split,
            config.features.acoustic_aggregation,
        )


def run(arguments: argparse.Namespace) -> None:
    if arguments.command == 'preprocess':
        _run_preprocess(arguments)
    elif arguments.command == 'train':
        from cognitive_impairments.preprocessing import resolve_device
        from cognitive_impairments.training import run_experiment_suite

        run_experiment_suite(
            arguments.data,
            arguments.config,
            arguments.output,
            arguments.seed,
            resolve_device(arguments.device),
        )
    elif arguments.command == 'paper-run':
        from cognitive_impairments.paper import run_paper_protocol
        from cognitive_impairments.preprocessing import resolve_device

        run_paper_protocol(
            dataset_path=arguments.data,
            config_path=arguments.config,
            output_directory=arguments.output,
            report_directory=arguments.report_output,
            device=resolve_device(arguments.device),
            resume=arguments.resume,
        )
    elif arguments.command == 'stats':
        from cognitive_impairments.statistics import save_dataset_statistics

        save_dataset_statistics(arguments.data, arguments.output)
    elif arguments.command == 'transcribe':
        from cognitive_impairments.transcription import transcribe_audio

        transcribe_audio(arguments.audio, arguments.output, arguments.model, arguments.language)
    elif arguments.command == 'explain':
        from cognitive_impairments.explanation import explain_cross_validation
        from cognitive_impairments.preprocessing import resolve_device

        explain_cross_validation(
            arguments.run,
            arguments.data,
            resolve_device(arguments.device),
        )
    elif arguments.command == 'predict' and arguments.prediction_mode == 'table':
        from cognitive_impairments.prediction import predict_table
        from cognitive_impairments.preprocessing import resolve_device

        predict_table(
            arguments.run,
            arguments.data,
            arguments.output,
            resolve_device(arguments.device),
            arguments.adresso_format,
        )
    elif arguments.command == 'predict' and arguments.prediction_mode == 'recording':
        from cognitive_impairments.prediction import predict_recording
        from cognitive_impairments.preprocessing import resolve_device

        predict_recording(
            arguments.run,
            arguments.audio,
            arguments.transcript,
            arguments.age,
            arguments.sex,
            arguments.output,
            resolve_device(arguments.device),
        )
    else:
        raise ValueError(f'Unsupported command: {arguments.command}')
