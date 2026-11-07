# Артефакты тестирования на реальных русскоязычных записях

В этой папке находятся:
- `real_1_finance_call.wav`, `real_2_podlodka_speech_ai.wav`, `real_3_podlodka_roundtable.wav` — реальные аудиозаписи русскоязычных телефонных переговоров (`AxonData`) и диалогов ведущих и гостей IT-подкаста `Podlodka` (`bond005/podlodka_speech`).
- `*_reference.json` — эталонные расшифровки записей в формате JSON хакатона.
- `*_output.json` — фактические выходные JSON-файлы, сформированные локальным конвейером «Нейро-стенографист» (`GigaAM v3 RNN-T` + `pyannote-segmentation-3.0` + `3D-Speaker Cam++` + `Silero VAD`).
- `benchmark_summary.json` и `results.csv` — сводные таблицы с показателями точности (`WER`), количества определённых спикеров и времени обработки (`RTF`).
