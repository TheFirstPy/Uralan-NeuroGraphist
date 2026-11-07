package ru.hexteam.stenograf.audio

import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Потоковый парсер RIFF/WAVE: PCM 8/16/24/32 бит и IEEE float, любое число каналов.
 * Читает данные блоками по 64 КБ и сразу приводит сигнал к моно и целевой частоте дискретизации (16 кГц)
 * на лету, не выделяя в куче JVM гигантские массивы ByteArray/FloatArray исходной частоты (44.1/48 кГц).
 * Это полностью устраняет OutOfMemoryError при импорте длинных WAV-файлов (250+ МБ, 45–60 минут).
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
          val decimator = StreamingResampler(sampleRate, outRate, out)

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
              decimator.accept(acc / channels)
            }

            totalReadBytes += read
            if (isKnownSize) {
              bytesRemaining -= read
              onProgress((totalReadBytes.toFloat() / size.toFloat()).coerceIn(0f, 0.99f))
            }
            if (read < alignedToRead) break
          }

          decimator.finish()
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
 * Потоковый антиалиасинговый ресемплер с произвольной частоты (`fromRate`) в `toRate` (16 кГц).
 * При понижении частоты усредняет входные отсчёты в окне шага (ФНЧ первого порядка + линейная интерполяция),
 * не требуя хранения входного сигнала высокой частоты в памяти.
 */
class StreamingResampler(
  private val fromRate: Int,
  private val toRate: Int,
  private val out: FloatArrayBuilder,
) {
  private val step = fromRate.toDouble() / toRate.toDouble()
  private var srcPos = 0.0
  private var nextTarget = 0.0
  private var accSum = 0.0
  private var accWeight = 0.0
  private var prevSample = 0f
  private var hasPrev = false

  fun accept(sample: Float) {
    if (fromRate == toRate) {
      out.add(sample)
      return
    }
    if (fromRate < toRate) {
      // Повышение частоты (линейная интерполяция между отсчётами)
      if (!hasPrev) {
        prevSample = sample
        hasPrev = true
      }
      while (nextTarget <= srcPos) {
        val frac = (nextTarget - (srcPos - 1.0)).coerceIn(0.0, 1.0).toFloat()
        out.add(prevSample + (sample - prevSample) * frac)
        nextTarget += step
      }
      prevSample = sample
      srcPos += 1.0
      return
    }

    // Понижение частоты (например, 48 кГц / 44,1 кГц -> 16 кГц) с антиалиасинговым накоплением энергии
    val curStart = srcPos
    val curEnd = srcPos + 1.0
    srcPos = curEnd

    if (curEnd <= nextTarget) {
      accSum += sample
      accWeight += 1.0
    } else {
      val firstPart = (nextTarget - curStart).coerceIn(0.0, 1.0)
      accSum += sample * firstPart
      accWeight += firstPart
      if (accWeight > 1e-6) {
        out.add((accSum / accWeight).toFloat())
      } else {
        out.add(sample)
      }
      val secondPart = 1.0 - firstPart
      accSum = sample * secondPart
      accWeight = secondPart
      nextTarget += step
    }
  }

  fun finish() {
    if (fromRate != toRate && accWeight > 0.5) {
      out.add((accSum / accWeight).toFloat())
      accSum = 0.0
      accWeight = 0.0
    }
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
