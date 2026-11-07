# Исходный код проекта «Нейро-стенографист»

## Структура каталога `src/`

* [`src/android/`](android/) — Android-приложение на Kotlin + Jetpack Compose (Material 3), использующее нативный движок `sherpa-onnx` (GigaAM v3 RNN-T, pyannote segmentation 3.0, 3D-Speaker Cam++, Silero VAD, GTCRN) и локальную LLM `Qwen3-0.6B` (`LiteRT-LM`).
* [`src/tools/eval/`](tools/eval/) — скрипты воспроизводимого тестирования и расчёта метрик точности транскрибации (WER), диаризации и скорости обработки (RTF) на реальных русскоязычных диалогах.

## Инструкция по сборке проекта с нуля

1. Скачайте офлайн-модели и нативный AAR в каталог `assets/` и `libs/`:
   ```bash
   cd src/android
   ./scripts/download_models.sh
   ```
2. Запустите модульные тесты и соберите готовый к установке APK:
   ```bash
   ./gradlew testDebugUnitTest assembleRelease
   ```
3. Готовый подписанный APK будет находиться по пути:
   ```
   src/android/app/build/outputs/apk/release/app-release.apk
   ```
