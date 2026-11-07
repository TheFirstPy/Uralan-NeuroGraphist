#!/usr/bin/env python3
"""
Стенд проверки конвейера «Уралан: Нейрографист» на реальном видеоподкасте
(«Дела», выпуск 67 — Дима Позов, Юля Ахмедова, Антон Шастун; фрагмент 16 мин).

Повторяет логику приложения (режим «Стандарт»):
  предобработка (ФВЧ 70 Гц + нормализация) -> Silero VAD -> GigaAM v3 RNN-T -> диаризация -> выравнивание.

Запуск:
  ~/hkvenv/bin/python src/tools/eval/podcast_eval.py asr       # распознавание + WER (кэшируется)
  ~/hkvenv/bin/python src/tools/eval/podcast_eval.py diar ...  # варианты диаризации
"""

import json
import math
import os
import re
import sys
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

SR = 16000
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS = os.path.join(ROOT, "src/android/app/src/main/assets/models")
OUT = os.path.join(ROOT, "artifacts/test-results/podcast")
WAV = os.path.join(OUT, "podcast_16k.wav")
REF = os.path.join(OUT, "reference.txt")
THREADS = 8


# ----------------------------------------------------------------------------- аудио
def preprocess(x: np.ndarray) -> np.ndarray:
    """Копия AudioPreprocessor.process: DC, ФВЧ Баттерворта 70 Гц, пик к 0.9 (не более +20 дБ)."""
    x = x.astype(np.float64) - x.mean()
    fc = 70.0
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * 0.7071)
    cosw = math.cos(w0)
    a0 = 1 + alpha
    b = np.array([(1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2]) / a0
    a = np.array([1.0, -2 * cosw / a0, (1 - alpha) / a0])
    from scipy.signal import lfilter  # noqa

    y = lfilter(b, a, x).astype(np.float32)
    peak = float(np.abs(y).max())
    gain = min(0.9 / peak, 10.0) if peak > 1e-6 else 1.0
    return (y * gain).astype(np.float32)


def load_audio():
    x, sr = sf.read(WAV, dtype="float32")
    assert sr == SR
    return preprocess(x)


# ----------------------------------------------------------------------------- VAD
def speech_windows(samples, max_sec=15.0, threshold=0.42, min_sil=0.30, pad=1600, merge_gap=0.25):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = f"{MODELS}/vad/silero_vad.onnx"
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_sil
    cfg.silero_vad.min_speech_duration = 0.18
    cfg.silero_vad.window_size = 512
    cfg.silero_vad.max_speech_duration = max_sec
    cfg.sample_rate = SR
    cfg.num_threads = 2
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    wins = []

    def drain():
        while not vad.empty():
            seg = vad.front
            s = max(0, seg.start - pad)
            e = min(len(samples), seg.start + len(seg.samples) + pad)
            if e - s >= SR // 6:
                wins.append((s, e))
            vad.pop()

    for pos in range(0, len(samples), 512):
        ch = samples[pos:pos + 512]
        if len(ch) < 512:
            ch = np.pad(ch, (0, 512 - len(ch)))
        vad.accept_waveform(ch)
        drain()
    vad.flush()
    drain()

    merged = []
    max_w = int(SR * max_sec)
    for s, e in sorted(wins):
        if merged and s - merged[-1][1] <= int(SR * merge_gap) and e - merged[-1][0] <= max_w:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


# ----------------------------------------------------------------------------- ASR
PUNCT = {".", ",", "?", "!", ":", ";", "—", "-", "»", "«", "..."}


def tokens_to_words(tokens, timestamps, ws_ms, we_ms):
    """Копия TranscriptAligner.tokensToWords."""
    words = []
    cur = ""
    wstart = -1
    last_t = ws_ms

    def flush(next_ms):
        nonlocal cur, wstart
        raw = cur.strip()
        if raw and raw not in PUNCT:
            s = max(wstart, ws_ms)
            est = max(s + 80, min(next_ms, last_t + 240))
            e = min(max(est, s + 40), max(s + 40, we_ms))
            words.append([raw, s, e])
        cur = ""
        wstart = -1

    for tok, ts in zip(tokens, timestamps):
        piece = tok.replace("▁", " ")
        t = min(max(ws_ms + int(round(ts * 1000)), ws_ms), we_ms)
        if not piece.strip():
            flush(t)
            continue
        tr = piece.strip()
        is_p = tr in PUNCT
        if piece.startswith(" ") and not is_p and cur:
            flush(t)
        if is_p:
            if cur:
                cur += tr
                last_t = max(last_t, t)
            elif words:
                words[-1][0] += tr
                words[-1][2] = max(words[-1][2], min(we_ms, t + 60))
        else:
            if wstart < 0:
                wstart = max(ws_ms, t - 40)
            cur = tr if not cur else cur + piece
            last_t = t
    flush(we_ms)
    return words


def make_recognizer():
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{MODELS}/asr/encoder.int8.onnx",
        decoder=f"{MODELS}/asr/decoder.onnx",
        joiner=f"{MODELS}/asr/joiner.onnx",
        tokens=f"{MODELS}/asr/tokens.txt",
        num_threads=THREADS,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )


def run_asr(samples, windows, rec=None):
    rec = rec or make_recognizer()
    words = []
    for s, e in windows:
        st = rec.create_stream()
        st.accept_waveform(SR, samples[s:e])
        rec.decode_stream(st)
        r = st.result
        ws_ms, we_ms = s * 1000 // SR, e * 1000 // SR
        if r.tokens:
            words += tokens_to_words(r.tokens, r.timestamps, ws_ms, we_ms)
        elif r.text.strip():
            words.append([r.text.strip(), ws_ms, we_ms])
    return words


# ----------------------------------------------------------------------------- WER
UNITS = {0: "", 1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять", 6: "шесть", 7: "семь", 8: "восемь",
         9: "девять", 10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
         15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать", 19: "девятнадцать"}
TENS = {2: "двадцать", 3: "тридцать", 4: "сорок", 5: "пятьдесят", 6: "шестьдесят", 7: "семьдесят",
        8: "восемьдесят", 9: "девяносто"}


def num_to_words(n: int) -> str:
    if n < 20:
        return UNITS[n] or "ноль"
    if n < 100:
        return (TENS[n // 10] + " " + UNITS[n % 10]).strip()
    return str(n)


def normalize_words(text: str):
    text = re.sub(r"\[[^\]]*\]", " ", text)           # [музыка], [смех]
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)    # таймкоды эталона
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\d+", lambda m: " " + num_to_words(int(m.group())) + " ", text)
    text = re.sub(r"[^\w\s-]", " ", text).replace("-", " ")
    return [w for w in text.split() if w]


def align(r, h):
    n, m = len(r), len(h)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = r[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            c = 0 if ri == h[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + c)
    # обратный проход
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            ops.append(("=" if r[i - 1] == h[j - 1] else "S", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("D", r[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", h[j - 1]))
            j -= 1
    return int(dp[n, m]), ops[::-1]


def ref_text_until(sec_limit):
    lines = open(REF, encoding="utf-8").read().splitlines()
    return " ".join(lines)


# ----------------------------------------------------------------------------- диаризация
def diar_config(threshold=0.5, num_clusters=-1, on=0.18, off=0.35):
    return sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=f"{MODELS}/diarization/segmentation.onnx"),
            num_threads=THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=f"{MODELS}/diarization/embedding.onnx", num_threads=THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=on,
        min_duration_off=off,
    )


def make_extractor():
    return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=f"{MODELS}/diarization/embedding.onnx", num_threads=THREADS))


def embed(ext, x):
    st = ext.create_stream()
    st.accept_waveform(SR, x)
    st.input_finished()
    if not ext.is_ready(st):
        return np.zeros(192, dtype=np.float32)
    v = np.array(ext.compute(st), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def cluster(embs, num_clusters, threshold, chunk_ids=None, penalty=0.22, weights=None):
    """Копия clusterNormalizedEmbeddings (центроидная связь, косинус)."""
    n = len(embs)
    if n <= 1:
        return [0] * n
    E = np.array(embs)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    clusters = [[i] for i in range(n)]
    cents = [E[i].copy() for i in range(n)]
    target = num_clusters if num_clusters > 0 else 1
    while len(clusters) > target:
        best = (1e9, -1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = 1 - float(cents[i] @ cents[j])
                if chunk_ids is not None and any(chunk_ids[a] == chunk_ids[b] for a in clusters[i] for b in clusters[j]):
                    d += penalty
                if d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        if num_clusters <= 0 and d > threshold:
            break
        clusters[i] += clusters[j]
        c = (E[clusters[i]] * w[clusters[i]][:, None]).sum(0)
        cents[i] = c / (np.linalg.norm(c) + 1e-9)
        del clusters[j], cents[j]
    lab = [0] * n
    for k, mem in enumerate(clusters):
        for m in mem:
            lab[m] = k
    return lab


def diar_app_standard(samples, threshold=0.5, num_clusters=-1):
    """Повтор runChunkedPyannoteDiarization (запись > 12 мин -> блоки по 5 мин + глобальная склейка)."""
    cfg = diar_config(threshold, num_clusters)
    single = SR * 720
    macro = SR * 300
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if len(samples) <= single:
        segs = sd.process(samples).sort_by_start_time()
        return [(int(s.start * 1000), int(s.end * 1000), s.speaker) for s in segs]
    chunks = []
    for c0 in range(0, len(samples), macro):
        c1 = min(len(samples), c0 + macro)
        off = c0 * 1000 // SR
        segs = sd.process(samples[c0:c1]).sort_by_start_time()
        chunks.append((c0, c1, [(off + int(s.start * 1000), off + int(s.end * 1000), s.speaker) for s in segs]))
    ext = make_extractor()
    keys, embs = [], []
    for ci, (c0, c1, sl) in enumerate(chunks):
        by = {}
        for s in sl:
            by.setdefault(s[2], []).append(s)
        for spk, lst in by.items():
            top = sorted(lst, key=lambda s: s[1] - s[0], reverse=True)[:3]
            parts = []
            for s, e, _ in top:
                a = min(max(s * SR // 1000, c0), c1)
                b = min(max(e * SR // 1000, a), c1)
                ln = min(b - a, SR * 4)
                if ln >= SR // 4:
                    parts.append(samples[a:a + ln])
            x = np.concatenate(parts) if parts else np.zeros(SR // 2, dtype=np.float32)
            keys.append((ci, spk))
            embs.append(embed(ext, x))
    lab = cluster(embs, num_clusters, threshold, chunk_ids=[k[0] for k in keys])
    km = {k: l for k, l in zip(keys, lab)}
    out = []
    for ci, (_, _, sl) in enumerate(chunks):
        for s, e, spk in sl:
            out.append((s, e, km[(ci, spk)]))
    return out


# ----------------------------------------------------------------------------- выравнивание
def best_speaker(s_ms, e_ms, slices):
    ov = {}
    for s, e, k in slices:
        o = min(e_ms, e) - max(s_ms, s)
        if o > 0:
            ov[k] = ov.get(k, 0) + o
    if ov:
        return max(ov.items(), key=lambda kv: kv[1])[0]
    mid = (s_ms + e_ms) // 2
    if not slices:
        return 0
    return min(slices, key=lambda sl: 0 if sl[0] <= mid <= sl[1] else min(abs(mid - sl[0]), abs(mid - sl[1])))[2]


def assign(words, slices):
    a = [[w[0], w[1], w[2], best_speaker(w[1], w[2], slices)] for w in words]
    for i in range(1, len(a) - 1):
        if (a[i - 1][3] == a[i + 1][3] and a[i][3] != a[i - 1][3]
                and a[i][1] - a[i - 1][2] < 300 and a[i + 1][1] - a[i][2] < 300 and a[i][2] - a[i][1] < 450):
            a[i][3] = a[i - 1][3]
    return a


def utterances(assigned):
    order = {}
    utts = []
    cur = []
    spk = None
    for w in assigned:
        if cur:
            pause = w[1] - cur[-1][2]
            ends = cur[-1][0].endswith((".", "!", "?"))
            if w[3] != spk or pause > 1500 or (pause > 850 and ends):
                utts.append((order.setdefault(spk, f"speaker_{len(order) + 1}"), cur))
                cur = []
        if not cur:
            spk = w[3]
        cur.append(w)
    if cur:
        utts.append((order.setdefault(spk, f"speaker_{len(order) + 1}"), cur))
    res = []
    for sid, ws in utts:
        res.append({"speaker_id": sid, "start_ms": ws[0][1], "end_ms": ws[-1][2],
                    "text": " ".join(x[0] for x in ws)})
    return res


def fmt(ms):
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{(ms // 100) % 10}"


def cmd_asr(variant="app"):
    samples = load_audio()
    t0 = time.time()
    if variant == "app":
        wins = speech_windows(samples)
    else:
        kw = json.loads(variant)
        wins = speech_windows(samples, **kw)
    words = run_asr(samples, wins)
    dt = time.time() - t0
    hyp = " ".join(w[0] for w in words)
    ref_w = normalize_words(ref_text_until(None))
    hyp_w = normalize_words(hyp)
    # эталон заканчивается раньше записи — обрезаем гипотезу по последним словам эталона
    dist, ops = align(ref_w, hyp_w)
    trail = 0
    for op in reversed(ops):
        if op[0] == "I":
            trail += 1
        else:
            break
    dist -= trail
    tag = "app" if variant == "app" else "var"
    json.dump(words, open(f"{OUT}/asr_words_{tag}.json", "w"), ensure_ascii=False)
    print(f"Окон: {len(wins)}, слов: {len(words)}, время {dt:.1f} с, RTF {dt / (len(samples) / SR):.3f}")
    print(f"WER (без хвоста после конца эталона, {trail} слов): {dist}/{len(ref_w)} = {100 * dist / len(ref_w):.2f}%")
    with open(f"{OUT}/asr_diff_{tag}.txt", "w", encoding="utf-8") as f:
        for op in ops[:len(ops) - trail]:
            if op[0] != "=":
                f.write(f"{op[0]}  {op[1]:<20} -> {op[2]}\n")


def cmd_diar(name, threshold, num_clusters, words_tag="app"):
    samples = load_audio()
    words = json.load(open(f"{OUT}/asr_words_{words_tag}.json"))
    t0 = time.time()
    fn = DIAR_VARIANTS[name]
    slices = fn(samples, threshold=threshold, num_clusters=num_clusters)
    dt = time.time() - t0
    a = assign(words, slices)
    utts = utterances(a)
    tot = {}
    for s, e, k in slices:
        tot[k] = tot.get(k, 0) + (e - s)
    print(f"[{name} thr={threshold} k={num_clusters}] время {dt:.1f} с, спикеров {len(tot)}: "
          + ", ".join(f"{k}:{v / 1000:.0f}с" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])))
    path = f"{OUT}/out_{name}_{threshold}_{num_clusters}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for u in utts:
            f.write(f"[{fmt(u['start_ms'])}-{fmt(u['end_ms'])}] {u['speaker_id']}: {u['text']}\n")
    json.dump({"transcript": utts}, open(path.replace(".txt", ".json"), "w"), ensure_ascii=False, indent=1)
    print(f"  реплик: {len(utts)} -> {path}")


DIAR_VARIANTS = {"app": diar_app_standard}

if __name__ == "__main__":
    if sys.argv[1] == "asr":
        cmd_asr(sys.argv[2] if len(sys.argv) > 2 else "app")
    elif sys.argv[1] == "diar":
        cmd_diar(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
