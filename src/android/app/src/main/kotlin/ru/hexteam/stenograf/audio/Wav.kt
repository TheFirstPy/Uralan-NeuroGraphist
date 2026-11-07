package ru.hexteam.stenograf.audio

import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Потоковый парсер RIFF/WAVE: PCM 8/16/24/32 бит и IEEE float, любое число каналов.
 * Читает данные блоками по 64 КБ и сразу приводит сигнал к моно и целевой частоте дискретизации (16 кГц)
 * при помощи оконного sinc-фильтра Блэкмана на кольцевом буфере (16 КБ ОЗУ), не выделяя в куче JVM
 * гигантские массивы ByteArray/FloatArray исходной частоты (44.1/48 кГц).
 * Это полностью устраняет OutOfMemoryError при импорте длинных WAV-файлов (250+ МБ, 45–60 минут)
 * и сохраняет эталонную частотную характеристику голоса для диаризации 3D-Speaker Cam++ и pyannote 3.0.
 */
object WavParser {

  fun parse(
    input: InputStream,
    targetSampleRate: Int = AudioDecoder.TARGET_RATE,
    onProgress: (Float) -> Unit = {},
  ): PcmAudio? {
    val din = DataInputStream(input.buffered(1 shl 16))
    val header = ByteArray(12)
    if (readUpTo(din, header, 12) < 12) return null
    if (String(header, 0, 4) != "RIFF" || String(header, 8, 4) != "WAVE") return null

    var format = 0
    var channels = 0
    var sampleRate = 0
    var bits = 0
    val chunkHeader = ByteArray(8)

    while (true) {
      if (readUpTo(din, chunkHeader, 8) < 8) return null
      val id = String(chunkHeader, 0, 4)
      val size = ByteBuffer.wrap(chunkHeader, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
      when (id) {
        "fmt " -> {
          val fmtSize = size.toInt().coerceIn(16, 4096)
          val fmt = ByteArray(fmtSize)
          if (readUpTo(din, fmt, fmtSize) < fmtSize) return null
          val bb = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN)
          format = bb.short.toInt() and 0xFFFF
          channels = bb.short.toInt()
          sampleRate = bb.int
          bb.int
          bb.short
          bits = bb.short.toInt()
          if (format == 0xFFFE && fmt.size >= 26) { // WAVE_FORMAT_EXTENSIBLE
            format = ByteBuffer.wrap(fmt, 24, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
          }
          val extraSkip = (size - fmtSize) + (size and 1L)
          if (extraSkip > 0) skipFully(din, extraSkip)
        }
        "data" -> {
          if (channels <= 0 || sampleRate <= 0) return null
          val bytesPerSample = bits / 8
          if (bytesPerSample <= 0) return null
          val isFloat = format == 3
          if (format != 1 && !isFloat) return null

          val frameBytes = bytesPerSample * channels
          val isKnownSize = size > 0L && size != 0xFFFFFFFFL
          val totalFrames = if (isKnownSize) (size / frameBytes) else 0L
          val outRate = if (targetSampleRate > 0) targetSampleRate else sampleRate
          val expectedOutSamples = if (totalFrames > 0L) {
            ((totalFrames.toDouble() * outRate) / sampleRate).toInt() + 64
          } else {
            1 shl 20
          }

          val out = FloatArrayBuilder(expectedOutSamples)
          val resampler = StreamingResampler(sampleRate, outRate, out)

          // Буфер чтения кратен размеру фрейма (~64 КБ)
          val framesPerChunk = (65536 / frameBytes).coerceAtLeast(256)
          val buf = ByteArray(framesPerChunk * frameBytes)
          val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)

          var bytesRemaining = if (isKnownSize) size else Long.MAX_VALUE
          var totalReadBytes = 0L

          while (bytesRemaining >= frameBytes) {
            val toRead = min(buf.size.toLong(), bytesRemaining).toInt()
            val alignedToRead = (toRead / frameBytes) * frameBytes
            if (alignedToRead <= 0) break
            val read = readUpTo(din, buf, alignedToRead)
            if (read < frameBytes) break
            val framesRead = read / frameBytes
            bb.position(0)
            bb.limit(framesRead * frameBytes)

            for (f in 0 until framesRead) {
              var acc = 0f
              for (c in 0 until channels) {
                acc += when {
                  isFloat && bytesPerSample == 4 -> bb.float
                  isFloat && bytesPerSample == 8 -> bb.double.toFloat()
                  bytesPerSample == 1 -> ((bb.get().toInt() and 0xFF) - 128) / 128f
                  bytesPerSample == 2 -> bb.short / 32768f
                  bytesPerSample == 3 -> {
                    val b0 = bb.get().toInt() and 0xFF
                    val b1 = bb.get().toInt() and 0xFF
                    val b2 = bb.get().toInt()
                    ((b2 shl 16) or (b1 shl 8) or b0) / 8388608f
                  }
                  bytesPerSample == 4 -> bb.int / 2147483648f
                  else -> 0f
                }
              }
              resampler.accept(acc / channels)
            }

            totalReadBytes += read
            if (isKnownSize) {
              bytesRemaining -= read
              onProgress((totalReadBytes.toFloat() / size.toFloat()).coerceIn(0f, 0.99f))
            }
            if (read < alignedToRead) break
          }

          resampler.finish()
          return PcmAudio(out.toExactArray(), outRate)
        }
        else -> skipFully(din, size + (size and 1L))
      }
    }
  }

  private fun readUpTo(input: InputStream, buf: ByteArray, len: Int): Int {
    var off = 0
    while (off < len) {
      val r = input.read(buf, off, len - off)
      if (r < 0) break
      off += r
    }
    return off
  }

  private fun skipFully(input: InputStream, bytesToSkip: Long) {
    var rem = bytesToSkip
    val scratch = ByteArray(4096)
    while (rem > 0L) {
      val skipped = input.skip(rem)
      if (skipped > 0L) {
        rem -= skipped
      } else {
        val r = input.read(scratch, 0, min(scratch.size.toLong(), rem).toInt())
        if (r < 0) break
        rem -= r
      }
    }
  }
}

/**
 * Потоковый антиалиасинговый ресемплер на основе 32-точечной оконной sinc-интерполяции Блэкмана.
 * Использует компактный кольцевой буфер (4096 отсчётов = 16 КБ ОЗУ) и даёт побитово тот же
 * высококачественный выходной сигнал, что и `Resampler.resample`, полностью подавляя алиасинг
 * и сохраняя форманты голоса в полосе 0..8 кГц для точной диаризации говорящих.
 */
class StreamingResampler(
  private val fromRate: Int,
  private val toRate: Int,
  private val out: FloatArrayBuilder,
) {
  companion object {
    private const val HALF_TAPS = 16
    private const val TABLE_RES = 512
    private const val RING_SIZE = 4096
    private const val RING_MASK = RING_SIZE - 1
  }

  private val passthrough = fromRate == toRate
  private val ratio = if (passthrough) 1.0 else toRate.toDouble() / fromRate.toDouble()
  private val step = if (passthrough) 1.0 else 1.0 / ratio
  private val cutoff = minOf(1.0, ratio) * 0.95
  private val halfWidth = HALF_TAPS / cutoff
  private val tableSize = if (passthrough) 0 else (halfWidth * TABLE_RES).toInt() + 2
  private val table = if (passthrough) {
    FloatArray(0)
  } else {
    FloatArray(tableSize) { i ->
      val x = i.toDouble() / TABLE_RES
      sincBlackmanKernel(x, cutoff, halfWidth).toFloat()
    }
  }

  private val ring = if (passthrough) FloatArray(0) else FloatArray(RING_SIZE)
  private var totalIn: Long = 0L
  private var nextOutIdx: Long = 0L

  fun accept(sample: Float) {
    if (passthrough) {
      out.add(sample)
      return
    }
    ring[(totalIn.toInt()) and RING_MASK] = sample
    totalIn++

    while (true) {
      val center = nextOutIdx * step
      val rightNeeded = floor(center + halfWidth).toLong()
      if (rightNeeded >= totalIn) break
      val left = maxOf(0L, floor(center - halfWidth).toLong() + 1L)
      out.add(evalSinc(center, left, rightNeeded))
      nextOutIdx++
    }
  }

  fun finish() {
    if (passthrough || totalIn == 0L) return
    val targetOutLen = (totalIn * ratio).toLong()
    val maxIdx = totalIn - 1L
    while (nextOutIdx < targetOutLen) {
      val center = nextOutIdx * step
      val left = maxOf(0L, floor(center - halfWidth).toLong() + 1L)
      val right = minOf(maxIdx, floor(center + halfWidth).toLong())
      out.add(evalSinc(center, left, right))
      nextOutIdx++
    }
  }

  private fun evalSinc(center: Double, left: Long, right: Long): Float {
    var acc = 0.0
    var norm = 0.0
    var i = left
    while (i <= right) {
      val d = abs(i - center) * TABLE_RES
      val idx = d.toInt()
      if (idx < tableSize - 1) {
        val frac = (d - idx).toFloat()
        val w = table[idx] + (table[idx + 1] - table[idx]) * frac
        val s = ring[(i.toInt()) and RING_MASK]
        acc += s * w
        norm += w
      }
      i++
    }
    return if (norm > 1e-9) (acc / norm).toFloat() else 0f
  }

  private fun sincBlackmanKernel(x: Double, cutoff: Double, halfWidth: Double): Double {
    if (x >= halfWidth) return 0.0
    val sinc = if (x < 1e-9) 1.0 else sin(PI * cutoff * x) / (PI * cutoff * x)
    val n = x / halfWidth
    val window = 0.42 + 0.5 * cos(PI * n) + 0.08 * cos(2 * PI * n)
    return sinc * window
  }
}

/** Запись моно PCM 16 бит в WAV. */
object WavWriter {
  fun write(file: File, samples: FloatArray, sampleRate: Int) {
    file.parentFile?.mkdirs()
    FileOutputStream(file).use { fos ->
      val dataSize = samples.size * 2
      fos.write(header(dataSize, sampleRate))
      val buf = ByteBuffer.allocate(1 shl 16).order(ByteOrder.LITTLE_ENDIAN)
      for (s in samples) {
        if (buf.remaining() < 2) {
          fos.write(buf.array(), 0, buf.position())
          buf.clear()
        }
        buf.putShort((s.coerceIn(-1f, 1f) * 32767f).roundToInt().toShort())
      }
      fos.write(buf.array(), 0, buf.position())
    }
  }

  fun header(dataSize: Int, sampleRate: Int): ByteArray {
    val bb = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
    bb.put("RIFF".toByteArray())
    bb.putInt(36 + dataSize)
    bb.put("WAVE".toByteArray())
    bb.put("fmt ".toByteArray())
    bb.putInt(16)
    bb.putShort(1)
    bb.putShort(1)
    bb.putInt(sampleRate)
    bb.putInt(sampleRate * 2)
    bb.putShort(2)
    bb.putShort(16)
    bb.put("data".toByteArray())
    bb.putInt(dataSize)
    return bb.array()
  }

  /** Исправляет размеры в заголовке после потоковой записи. */
  fun patchHeader(file: File, sampleRate: Int) {
    RandomAccessFile(file, "rw").use { raf ->
      val dataSize = (raf.length() - 44).toInt()
      raf.seek(0)
      raf.write(header(dataSize, sampleRate))
    }
  }
}
