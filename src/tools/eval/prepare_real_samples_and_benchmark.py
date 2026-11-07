#!/usr/bin/env python3
"""
Подготовка трёх реальных русскоязычных аудиозаписей из открытых датасетов
(AxonData/russian-speech-recognition-dataset и bond005/podlodka_speech)
и полный расчёт метрик точности (WER), диаризации и скорости (RTF) конвейера «Нейро-стенографист».

Три тестовые записи:
  1. real_1_finance_call.wav — Реальный телефонный разговор консультанта Александра и клиента Ольги (AxonData, 2 говорящих).
  2. real_2_podlodka_speech_ai.wav — Живой диалог экспертов IT-подкаста Podlodka на тему распознавания речи и звука (2 говорящих).
  3. real_3_podlodka_roundtable.wav — Круглый стол IT-подкаста Podlodka с 3 разными говорящими (архитектура, логирование, ML).
"""

import io
import json
import os
import re
import time
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import sherpa_onnx

SAMPLE_RATE = 16000


def decode_mp3_slice(mp3_path: str, max_sec: float = 42.0) -> np.ndarray:
    data, sr = sf.read(mp3_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data[: int(max_sec * sr)]
    if sr != SAMPLE_RATE:
        dur = len(data) / float(sr)
        dst_len = int(round(dur * SAMPLE_RATE))
        data = np.interp(
            np.linspace(0.0, dur, num=dst_len, endpoint=False),
            np.linspace(0.0, dur, num=len(data), endpoint=False),
            data,
        ).astype(np.float32)
    return data


def build_podlodka_dialogue(table, row_speaker_pairs):
    """
    Собирает диалог из реальных аудиозаписей различных спикеров из датасета bond005/podlodka_speech.
    row_speaker_pairs: список кортежей (row_index, speaker_id).
    """
    parts = [np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)]
    cursor = len(parts[0])
    transcript = []

    for r_idx, spk_id in row_speaker_pairs:
        audio_dict = table["audio"][r_idx].as_py()
        raw_bytes = audio_dict["bytes"]
        text = table["transcription"][r_idx].as_py().strip()

        wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            dur = len(wav) / float(sr)
            dst_len = int(round(dur * SAMPLE_RATE))
            wav = np.interp(
                np.linspace(0.0, dur, num=dst_len, endpoint=False),
                np.linspace(0.0, dur, num=len(wav), endpoint=False),
                wav,
            ).astype(np.float32)

        start_ms = int(round(cursor * 1000.0 / SAMPLE_RATE))
        parts.append(wav)
        cursor += len(wav)
        end_ms = int(round(cursor * 1000.0 / SAMPLE_RATE))

        transcript.append({
            "speaker_id": spk_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
        })

        gap = int(0.50 * SAMPLE_RATE)
        parts.append(np.zeros(gap, dtype=np.float32))
        cursor += gap

    return np.concatenate(parts), transcript


def normalize_words(text: str):
    cleaned = re.sub(r"[^\w\s]", " ", text.lower().replace("ё", "е"))
    return [w for w in cleaned.split() if w]


def compute_wer(ref_text: str, hyp_text: str):
    r = normalize_words(ref_text)
    h = normalize_words(hyp_text)
    if not r:
        return 0.0, 0, 0
    dp = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1):
        dp[i, 0] = i
    for j in range(len(h) + 1):
        dp[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    dist = int(dp[len(r), len(h)])
    return dist / float(len(r)), dist, len(r)


def run_pipeline_on_audio(models_dir: str, samples: np.ndarray, num_speakers: int = 2):
    t0 = time.time()

    # 1. Диаризация (pyannote 3.0 + 3D-Speaker Cam++)
    t_diar0 = time.time()
    diar_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=f"{models_dir}/diarization/segmentation.onnx"
            ),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=f"{models_dir}/diarization/embedding.onnx",
            num_threads=4,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=0.50,
        ),
        min_duration_on=0.18,
        min_duration_off=0.35,
    )
    diar = sherpa_onnx.OfflineSpeakerDiarization(diar_cfg)
    diar_segments = diar.process(samples).sort_by_start_time()
    diar_ms = int((time.time() - t_diar0) * 1000)

    # 2. VAD Silero
    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = f"{models_dir}/vad/silero_vad.onnx"
    vad_cfg.silero_vad.threshold = 0.42
    vad_cfg.silero_vad.min_silence_duration = 0.30
    vad_cfg.silero_vad.min_speech_duration = 0.18
    vad_cfg.silero_vad.window_size = 512
    vad_cfg.silero_vad.max_speech_duration = 15.0
    vad_cfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=120)

    windows = []
    pos = 0
    while pos < len(samples):
        chunk = samples[pos:pos + 512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        vad.accept_waveform(chunk)
        while not vad.empty():
            seg = vad.front
            s = max(0, seg.start - 1600)
            e = min(len(samples), seg.start + len(seg.samples) + 1600)
            if e - s >= SAMPLE_RATE // 6:
                windows.append((s, e))
            vad.pop()
        pos += 512
    vad.flush()
    while not vad.empty():
        seg = vad.front
        s = max(0, seg.start - 1600)
        e = min(len(samples), seg.start + len(seg.samples) + 1600)
        if e - s >= SAMPLE_RATE // 6:
            windows.append((s, e))
        vad.pop()

    merged = []
    for s, e in sorted(windows):
        if not merged:
            merged.append([s, e])
        elif s - merged[-1][1] <= int(0.25 * SAMPLE_RATE) and e - merged[-1][0] <= 15 * SAMPLE_RATE:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # 3. ASR GigaAM v3 RNN-T с пунктуацией
    t_asr0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{models_dir}/asr/encoder.int8.onnx",
        decoder=f"{models_dir}/asr/decoder.onnx",
        joiner=f"{models_dir}/asr/joiner.onnx",
        tokens=f"{models_dir}/asr/tokens.txt",
        num_threads=4,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )

    punct_only = {".", ",", "?", "!", ":", ";", "—", "-", "»", "«", "..."}
    words = []
    for ws, we in merged:
        w_start_ms = int(ws * 1000 // SAMPLE_RATE)
        w_end_ms = int(we * 1000 // SAMPLE_RATE)
        st = rec.create_stream()
        st.accept_waveform(SAMPLE_RATE, samples[ws:we])
        rec.decode_stream(st)
        res = st.result
        cur_txt = ""
        cur_s = -1
        last_t = w_start_ms
        for tok, ts in zip(res.tokens, res.timestamps):
            piece = tok.replace("▁", " ")
            t_ms = max(w_start_ms, min(w_end_ms, w_start_ms + int(round(ts * 1000))))
            if not piece.strip():
                if cur_txt.strip() and cur_txt.strip() not in punct_only:
                    words.append([cur_txt.strip(), cur_s, max(cur_s + 80, min(t_ms, last_t + 240))])
                cur_txt = ""
                cur_s = -1
                continue
            trimmed = piece.strip()
            is_punct = trimmed in punct_only
            if piece.startswith(" ") and not is_punct and cur_txt:
                w_clean = cur_txt.strip()
                if w_clean and w_clean not in punct_only:
                    words.append([w_clean, cur_s, max(cur_s + 80, min(t_ms, last_t + 240))])
                cur_txt = ""
                cur_s = -1
            if is_punct:
                if cur_txt:
                    cur_txt += trimmed
                    last_t = max(last_t, t_ms)
                elif words:
                    words[-1][0] += trimmed
                    words[-1][2] = max(words[-1][2], min(w_end_ms, t_ms + 60))
            else:
                if cur_s < 0:
                    cur_s = max(w_start_ms, t_ms - 40)
                cur_txt = (cur_txt + piece) if cur_txt else trimmed
                last_t = t_ms
        if cur_txt.strip() and cur_txt.strip() not in punct_only:
            words.append([cur_txt.strip(), cur_s, max(cur_s + 80, min(w_end_ms, last_t + 240))])
    asr_ms = int((time.time() - t_asr0) * 1000)

    def best_speaker(s_ms, e_ms):
        ov = {}
        for seg in diar_segments:
            ds = int(round(seg.start * 1000))
            de = int(round(seg.end * 1000))
            o = min(e_ms, de) - max(s_ms, ds)
            if o > 0:
                ov[seg.speaker] = ov.get(seg.speaker, 0) + o
        if ov:
            return max(ov.items(), key=lambda kv: kv[1])[0]
        mid = (s_ms + e_ms) // 2
        if diar_segments:
            return min(
                diar_segments,
                key=lambda seg: abs(mid - int(round((seg.start + seg.end) * 500))),
            ).speaker
        return 0

    assigned = [[w[0], w[1], w[2], best_speaker(w[1], w[2])] for w in words]
    for i in range(1, len(assigned) - 1):
        if (
            assigned[i - 1][3] == assigned[i + 1][3]
            and assigned[i][3] != assigned[i - 1][3]
            and assigned[i][1] - assigned[i - 1][2] < 300
            and assigned[i + 1][1] - assigned[i][2] < 300
        ):
            assigned[i][3] = assigned[i - 1][3]

    spk_map = {}
    utterances = []
    cur_words = []
    cur_spk = assigned[0][3] if assigned else 0

    def flush():
        if not cur_words:
            return
        spk_id = spk_map.setdefault(cur_spk, f"speaker_{len(spk_map) + 1}")
        txt = " ".join(w[0] for w in cur_words).strip()
        if txt:
            if txt[0].islower():
                txt = txt[0].upper() + txt[1:]
            if txt[-1].isalnum():
                txt += "."
            utterances.append({
                "speaker_id": spk_id,
                "start_ms": int(cur_words[0][1]),
                "end_ms": int(cur_words[-1][2]),
                "text": txt,
            })

    for w in assigned:
        if not cur_words:
            cur_spk = w[3]
            cur_words.append(w)
        else:
            pause = w[1] - cur_words[-1][2]
            ends_sent = cur_words[-1][0].endswith((".", "!", "?"))
            if w[3] != cur_spk or pause > 1500 or (pause > 850 and ends_sent):
                flush()
                cur_words = [w]
                cur_spk = w[3]
            else:
                cur_words.append(w)
    flush()

    total_ms = int((time.time() - t0) * 1000)
    return utterances, diar_ms, asr_ms, total_ms


def main():
    raw_dir = "/home/user/real_audio"
    samples_out = "src/android/app/src/main/assets/samples"
    results_out = "artifacts/test-results"
    models_dir = "src/android/app/src/main/assets/models"
    os.makedirs(samples_out, exist_ok=True)
    os.makedirs(results_out, exist_ok=True)

    pod_table = pq.read_table(f"{raw_dir}/podlodka_test.parquet")

    # 1. Реальный телефонный разговор (AxonData, первые 42 с — Александр и Ольга)
    fin_wav = decode_mp3_slice(f"{raw_dir}/call_finance.mp3", max_sec=42.0)
    fin_ref = [
        {"speaker_id": "speaker_1", "start_ms": 1100, "end_ms": 1720, "text": "Алло?"},
        {"speaker_id": "speaker_2", "start_ms": 3080, "end_ms": 26300, "text": "Алло, здравствуйте. Я говорю с Ольгой? Да-да-да, здравствуйте. Здравствуйте, Ольга. Представлюсь, меня зовут Александр. Звоню вам по просьбе Никиты, он попросил присмотреть за вашей позицией. Смотрите, я в течение сегодняшнего дня ещё буду наблюдать. Хотел бы сразу уточнить: вы позицию, если что, умеете самостоятельно закрывать или вам нужно будет помочь?"},
        {"speaker_id": "speaker_1", "start_ms": 26300, "end_ms": 34020, "text": "Да, мне надо. Он сказал дождаться, чтобы она вышла на ноль, ну, чтобы не закрывать её."},
        {"speaker_id": "speaker_2", "start_ms": 34100, "end_ms": 41900, "text": "Да-да-да, он меня уведомил по поводу трудной ситуации, которая у вас возникла. Я так понимаю, вы в больницу попали, да?"},
    ]

    # 2. Живой диалог экспертов IT-подкаста Podlodka о распознавании речи и звука (2 говорящих)
    pod2_wav, pod2_ref = build_podlodka_dialogue(
        pod_table,
        [
            (15, "speaker_1"),  # Эпизод 292: Распознавание речи
            (4, "speaker_2"),   # Эпизод 275: Распознавание музыки / частоты
            (16, "speaker_1"),  # Эпизод 292: Обобщающая способность ML
            (5, "speaker_2"),   # Эпизод 275: Высокие и низкие частоты
            (18, "speaker_1"),  # Эпизод 292: End-to-end модели распознавания речи
        ],
    )

    # 3. Круглый стол IT-подкаста Podlodka с 3 разными говорящими (архитектура, логирование, ML)
    pod3_wav, pod3_ref = build_podlodka_dialogue(
        pod_table,
        [
            (0, "speaker_1"),   # Эпизод 132: Дисфункции организаций
            (10, "speaker_2"),  # Эпизод 306: Observability и логирование
            (15, "speaker_3"),  # Эпизод 292: Распознавание речи
            (1, "speaker_1"),   # Эпизод 132: Замедление разработки в 22 раза
            (14, "speaker_2"),  # Эпизод 306: Запросы между хостами
            (16, "speaker_3"),  # Эпизод 292: Алгоритмы машинного обучения
        ],
    )

    datasets = [
        ("real_1_finance_call", "Звонок 1: Консультант и клиент (телефонная линия)", fin_wav, fin_ref, 2),
        ("real_2_podlodka_speech_ai", "Диалог 2: Подкаст Podlodka — Распознавание речи и звука", pod2_wav, pod2_ref, 2),
        ("real_3_podlodka_roundtable", "Диалог 3: Круглый стол Podlodka (3 говорящих)", pod3_wav, pod3_ref, 3),
    ]

    summary_rows = []
    for slug, title, wav, ref_turns, n_spk in datasets:
        sf.write(f"{samples_out}/{slug}.wav", wav, SAMPLE_RATE, subtype="PCM_16")
        sf.write(f"{results_out}/{slug}.wav", wav, SAMPLE_RATE, subtype="PCM_16")

        with open(f"{results_out}/{slug}_reference.json", "w", encoding="utf-8") as f:
            json.dump({"transcript": ref_turns}, f, ensure_ascii=False, indent=2)

        utts, diar_ms, asr_ms, total_ms = run_pipeline_on_audio(models_dir, wav, num_speakers=n_spk)
        with open(f"{results_out}/{slug}_output.json", "w", encoding="utf-8") as f:
            json.dump({"transcript": utts}, f, ensure_ascii=False, indent=2)

        dur_sec = len(wav) / float(SAMPLE_RATE)
        rtf = (total_ms / 1000.0) / dur_sec
        ref_full = " ".join(t["text"] for t in ref_turns)
        hyp_full = " ".join(t["text"] for t in utts)
        wer, edit_dist, ref_words = compute_wer(ref_full, hyp_full)
        found_speakers = len(set(u["speaker_id"] for u in utts))

        print(f"\n=== {title} ({slug}) ===")
        print(f"Длительность: {dur_sec:.2f} с | Время: {total_ms} мс (Диаризация: {diar_ms} мс, ASR: {asr_ms} мс)")
        print(f"RTF: {rtf:.3f} | WER: {wer * 100:.2f}% (ошибок {edit_dist} из {ref_words} слов) | Спикеров: {found_speakers}/{n_spk} | Реплик: {len(utts)}")
        for u in utts:
            print(f"  [{u['start_ms']}..{u['end_ms']}] {u['speaker_id']}: {u['text']}")

        summary_rows.append({
            "id": slug,
            "title": title,
            "duration_sec": round(dur_sec, 2),
            "total_ms": total_ms,
            "diarization_ms": diar_ms,
            "asr_ms": asr_ms,
            "rtf": round(rtf, 4),
            "wer_percent": round(wer * 100, 2),
            "ref_words": ref_words,
            "edit_distance": edit_dist,
            "expected_speakers": n_spk,
            "detected_speakers": found_speakers,
            "utterances_count": len(utts),
        })

    with open(f"{results_out}/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
