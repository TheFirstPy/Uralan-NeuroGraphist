#!/usr/bin/env bash
# Скачивает все модели и нативные библиотеки, необходимые для сборки APK.
# После выполнения все модели лежат в app/src/main/assets и попадают в APK,
# поэтому приложение работает полностью офлайн сразу после установки.
#
# Использование:  ./scripts/download_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="$ROOT/app/src/main/assets"
LIBS="$ROOT/app/libs"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SHERPA_VERSION="2.0.1"
GH="https://github.com/k2-fsa/sherpa-onnx/releases/download"
HF="https://huggingface.co"

mkdir -p "$ASSETS/models/asr" "$ASSETS/models/diarization" "$ASSETS/models/vad" "$ASSETS/models/denoise" "$ASSETS/llm" "$LIBS"

fetch() { # url dst
  if [ -s "$2" ]; then echo "✓ уже есть: $(basename "$2")"; return; fi
  echo "↓ $(basename "$2")"
  curl -fL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

# 1. sherpa-onnx (ONNX Runtime + JNI) для Android
fetch "$GH/v$SHERPA_VERSION/sherpa-onnx-$SHERPA_VERSION.aar" "$LIBS/sherpa-onnx-$SHERPA_VERSION.aar"

# 2. ASR: GigaAM v3 RNN-T с пунктуацией и заглавными буквами (SberDevices), int8
ASR_NAME="sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"
if [ ! -s "$ASSETS/models/asr/encoder.int8.onnx" ]; then
  fetch "$GH/asr-models/$ASR_NAME.tar.bz2" "$TMP/asr.tar.bz2"
  tar xjf "$TMP/asr.tar.bz2" -C "$TMP"
  cp "$TMP/$ASR_NAME/encoder.int8.onnx" "$TMP/$ASR_NAME/decoder.onnx" \
     "$TMP/$ASR_NAME/joiner.onnx" "$TMP/$ASR_NAME/tokens.txt" "$ASSETS/models/asr/"
fi

# 3. Диаризация: сегментация pyannote 3.0 + эмбеддинги говорящих
if [ ! -s "$ASSETS/models/diarization/segmentation.onnx" ]; then
  fetch "$GH/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" "$TMP/seg.tar.bz2"
  tar xjf "$TMP/seg.tar.bz2" -C "$TMP"
  cp "$TMP/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" "$ASSETS/models/diarization/segmentation.onnx"
fi
fetch "$GH/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx" \
      "$ASSETS/models/diarization/embedding.onnx"

# 4. VAD: Silero
fetch "$GH/asr-models/silero_vad.onnx" "$ASSETS/models/vad/silero_vad.onnx"

# 5. Шумоподавление: GTCRN (опционально включается в настройках)
fetch "$GH/speech-enhancement-models/gtcrn_simple.onnx" "$ASSETS/models/denoise/gtcrn_simple.onnx"

# 6. Локальная LLM: Qwen3-0.6B (INT4) для LiteRT-LM
fetch "$HF/litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm" \
      "$ASSETS/llm/qwen3-0.6b.litertlm"

echo
echo "Готово. Содержимое assets:"
du -sh "$ASSETS"/models/* "$ASSETS"/llm
