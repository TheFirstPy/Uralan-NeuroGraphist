package ru.hexteam.stenograf.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Ресемплер на основе оконной sinc-интерполяции (окно Блэкмана).
 * При понижении частоты автоматически работает как антиалиасинговый ФНЧ,
 * что заметно лучше линейной интерполяции для качества распознавания.
 */
object Resampler {
  private const val HALF_TAPS = 16

  fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
    if (fromRate == toRate || input.isEmpty()) return input
    val ratio = toRate.toDouble() / fromRate
    val outLen = (input.size * ratio).toLong().toInt()
    val out = FloatArray(outLen)
    val cutoff = minOf(1.0, ratio) * 0.95 // доля от частоты Найквиста исходного сигнала
    val step = 1.0 / ratio
    val halfWidth = HALF_TAPS / cutoff

    // Таблица ядра для ускорения: 512 под-отсчётов на единицу
    val tableRes = 512
    val tableSize = (halfWidth * tableRes).toInt() + 2
    val table = FloatArray(tableSize) { i ->
      val x = i.toDouble() / tableRes
      kernel(x, cutoff, halfWidth).toFloat()
    }

    for (o in 0 until outLen) {
      val center = o * step
      val left = maxOf(0, floor(center - halfWidth).toInt() + 1)
      val right = minOf(input.size - 1, floor(center + halfWidth).toInt())
      var acc = 0.0
      var norm = 0.0
      for (i in left..right) {
        val d = abs(i - center) * tableRes
        val idx = d.toInt()
        if (idx >= tableSize - 1) continue
        val frac = (d - idx).toFloat()
        val w = table[idx] + (table[idx + 1] - table[idx]) * frac
        acc += input[i] * w
        norm += w
      }
      out[o] = if (norm > 1e-9) (acc / norm).toFloat() else 0f
    }
    return out
  }

  private fun kernel(x: Double, cutoff: Double, halfWidth: Double): Double {
    if (x >= halfWidth) return 0.0
    val sinc = if (x < 1e-9) 1.0 else sin(PI * cutoff * x) / (PI * cutoff * x)
    val n = x / halfWidth // 0..1
    val window = 0.42 + 0.5 * cos(PI * n) + 0.08 * cos(2 * PI * n)
    return sinc * window
  }
}

/** Базовая предобработка: удаление постоянной составляющей, ФВЧ 70 Гц, нормализация громкости. */
object AudioPreprocessor {

  data class Stats(val peak: Float, val rms: Float, val gain: Float)

  fun process(samples: FloatArray, sampleRate: Int): Stats {
    if (samples.isEmpty()) return Stats(0f, 0f, 1f)
    // 1) DC offset
    var mean = 0.0
    for (s in samples) mean += s
    mean /= samples.size
    // 2) ФВЧ Баттерворта 2-го порядка (biquad) — убирает гул, удары, ветер
    val fc = 70.0
    val w0 = 2 * PI * fc / sampleRate
    val alpha = sin(w0) / (2 * 0.7071)
    val cosw = cos(w0)
    val a0 = 1 + alpha
    val b0 = ((1 + cosw) / 2) / a0
    val b1 = (-(1 + cosw)) / a0
    val b2 = ((1 + cosw) / 2) / a0
    val a1 = (-2 * cosw) / a0
    val a2 = (1 - alpha) / a0
    var x1 = 0.0; var x2 = 0.0; var y1 = 0.0; var y2 = 0.0
    var peak = 0f
    var energy = 0.0
    for (i in samples.indices) {
      val x = samples[i] - mean
      val y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
      x2 = x1; x1 = x; y2 = y1; y1 = y
      val f = y.toFloat()
      samples[i] = f
      val a = abs(f)
      if (a > peak) peak = a
      energy += f * f
    }
    val rms = sqrt(energy / samples.size).toFloat()
    // 3) Нормализация: пик к 0.9, но усиление не более 20 дБ (не раздуваем шум)
    val gain = if (peak > 1e-6f) minOf(0.9f / peak, 10f) else 1f
    if (abs(gain - 1f) > 0.01f) for (i in samples.indices) samples[i] *= gain
    return Stats(peak, rms, gain)
  }
}

/**
 * Оценка медианной частоты основного тона голоса (F0, 72..360 Гц) методом нормированной автокорреляции
 * с подавлением субгармоник (удвоения периода). Используется для повышения точности разделения
 * мужских и женских голосов, а также разных регистров тембра при диаризации.
 */
object VoicePitchAnalyzer {
  private val LN2 = ln(2.0).toFloat()

  fun computeMedianF0Hz(
    samples: FloatArray,
    fromIdx: Int = 0,
    toIdx: Int = samples.size,
    sampleRate: Int = 16_000,
  ): Float {
    val start = fromIdx.coerceIn(0, samples.size)
    val end = toIdx.coerceIn(start, samples.size)
    val frameLen = (sampleRate * 0.032f).toInt() // 32 мс (512 отсчётов при 16 кГц)
    val hop = frameLen / 2
    if (end - start < frameLen) return 0f

    val minLag = (sampleRate / 360).coerceAtLeast(1)
    val maxLag = (sampleRate / 72).coerceAtMost(frameLen - 1)
    val f0Values = ArrayList<Float>()
    val centered = FloatArray(frameLen)

    var pos = start
    while (pos + frameLen <= end) {
      var mean = 0f
      for (k in 0 until frameLen) mean += samples[pos + k]
      mean /= frameLen

      var r0 = 0f
      for (k in 0 until frameLen) {
        val v = samples[pos + k] - mean
        centered[k] = v
        r0 += v * v
      }
      if (r0 / frameLen >= 3e-4f) {
        var bestLag = minLag
        var maxPeak = -1f
        val acByLag = FloatArray(maxLag + 1)
        for (lag in minLag..maxLag) {
          var sum = 0f
          val limit = frameLen - lag
          for (k in 0 until limit) {
            sum += centered[k] * centered[k + lag]
          }
          val normAc = sum / (r0 + 1e-9f)
          acByLag[lag] = normAc
          if (normAc > maxPeak) {
            maxPeak = normAc
            bestLag = lag
          }
        }
        if (maxPeak >= 0.42f) {
          val halfLag = bestLag / 2
          val chosenLag = if (halfLag >= minLag && acByLag[halfLag] >= 0.80f * maxPeak) {
            halfLag
          } else {
            bestLag
          }
          f0Values.add(sampleRate.toFloat() / chosenLag)
        }
      }
      pos += hop
    }

    if (f0Values.isEmpty()) return 0f
    f0Values.sort()
    val mid = f0Values.size / 2
    return if (f0Values.size % 2 == 1) {
      f0Values[mid]
    } else {
      0.5f * (f0Values[mid - 1] + f0Values[mid])
    }
  }

  /** Октавное расстояние между двумя оценками основного тона (до +0.25 к косинусному расстоянию). */
  fun f0Distance(f0A: Float, f0B: Float): Float {
    if (f0A <= 50f || f0B <= 50f) return 0f
    val octDiff = abs(ln((f0A / f0B).toDouble()).toFloat() / LN2)
    return min(0.25f, octDiff * 0.18f)
  }
}
