from __future__ import annotations

import pytest
from torch import Tensor, device, tensor

from cognitive_impairments.preprocessing import cognitive_impairment_label
from cognitive_impairments.preprocessing import resolve_device


@pytest.mark.parametrize(
    ('diagnosis', 'expected'),
    [
        ('Control', 0),
        ('PossibleAD', 1),
        ('ProbableAD', 1),
        ('MCI', 1),
        ('Memory', 1),
        ('Vascular', 1),
        ('Other', 1),
    ],
)
def test__cognitive_impairment_label__all_supported_diagnoses(
    diagnosis: str,
    expected: int,
) -> None:
    """
    Arrange: задано исходное диагностическое значение Pitt.
    Act: значение преобразуется в бинарную цель исследования.
    Assert: только Control относится к отрицательному классу.
    """
    result = cognitive_impairment_label(diagnosis)

    assert result == expected


def test__cognitive_impairment_label__unsupported_diagnosis() -> None:
    """
    Arrange: задано неизвестное диагностическое значение.
    Act: выполняется преобразование в бинарную цель.
    Assert: неизвестное значение не принимается как положительный класс молча.
    """
    with pytest.raises(ValueError):
        cognitive_impairment_label('Unknown')


def test__resolve_device__explicit_cpu() -> None:
    """
    Arrange: явно выбран процессор для диагностического запуска.
    Act: определяется вычислительное устройство.
    Assert: возвращается процессор независимо от доступности CUDA.
    """
    result = resolve_device('cpu')

    assert result == device('cpu')


def test__resolve_device__cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Arrange: CUDA недоступна при выборе основного вычислительного устройства.
    Act: выполняется определение устройства.
    Assert: запуск завершается понятной ошибкой без неявного перехода на процессор.
    """
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)

    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        resolve_device('auto')


def test__resolve_device__cuda_startup_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Arrange: CUDA доступна и тестовая операция выполняется успешно.
    Act: определяется автоматическое вычислительное устройство.
    Assert: возвращается устройство CUDA.
    """
    def create_test_tensor(*args: object, **kwargs: object) -> Tensor:
        return tensor([1])

    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.ones', create_test_tensor)

    result = resolve_device('auto')

    assert result == device('cuda')


def test__resolve_device__cuda_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Arrange: CUDA обнаружена, но тестовая операция завершается ошибкой.
    Act: определяется автоматическое вычислительное устройство.
    Assert: исходная ошибка оборачивается понятным сообщением о настройке CUDA.
    """
    def fail_test_tensor(*args: object, **kwargs: object) -> Tensor:
        raise RuntimeError('incompatible CUDA build')

    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.ones', fail_test_tensor)

    with pytest.raises(RuntimeError, match='CUDA failed its startup check'):
        resolve_device('auto')
