package ru.hexteam.stenograf.audio

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Декодированный сигнал: моно, float в диапазоне [-1, 1]. */
class PcmAudio(val samples: FloatArray, val sampleRate: Int) {
  val durationMs: Long get() = samples.size * 1000L / sampleRate
}

/**
 * Потоковый декодер аудио любых форматов, поддерживаемых Android (WAV, MP3, M4A/AAC, OGG/Opus, FLAC, AMR, 3GP…).
 * WAV разбирается потоково блоками по 64 КБ, остальные форматы — через MediaCodec.
 * Сведение в моно и ресемплинг до 16 кГц выполняются на лету в процессе чтения пакетов,
 * что исключает переполнение памяти (OOM) на больших файлах (250–500 МБ, 45–90 минут).
 */
object AudioDecoder {
  const val TARGET_RATE = 16_000

  fun decodeToMono16k(context: Context, uri: Uri, onProgress: (Float) -> Unit = {}): PcmAudio {
    val raw = tryDecodeWav(context, uri, onProgress) ?: decodeWithMediaCodec(context, uri, onProgress)
    onProgress(1f)
    return if (raw.sampleRate == TARGET_RATE) raw
    else PcmAudio(Resampler.resample(raw.samples, raw.sampleRate, TARGET_RATE), TARGET_RATE)
  }

  fun decodeFileToMono16k(context: Context, file: File): PcmAudio =
    decodeToMono16k(context, Uri.fromFile(file))

  // ---------------------------------------------------------------- WAV

  private fun tryDecodeWav(context: Context, uri: Uri, onProgress: (Float) -> Unit): PcmAudio? {
    return try {
      openStream(context, uri)?.use { input ->
        WavParser.parse(input, targetSampleRate = TARGET_RATE, onProgress = onProgress)
      }
    } catch (_: Throwable) {
      null
    }
  }

  private fun openStream(context: Context, uri: Uri): InputStream? =
    if (uri.scheme == "file") FileInputStream(File(uri.path!!)) else context.contentResolver.openInputStream(uri)

  // ---------------------------------------------------------- MediaCodec

  private fun decodeWithMediaCodec(context: Context, uri: Uri, onProgress: (Float) -> Unit): PcmAudio {
    val extractor = MediaExtractor()
    var tempCopy: File? = null
    try {
      if (uri.scheme == "file") {
        extractor.setDataSource(uri.path!!)
      } else {
        try {
          extractor.setDataSource(context, uri, null)
        } catch (_: Throwable) {
          val cached = File(context.cacheDir, "import_${System.currentTimeMillis()}.tmp")
          openStream(context, uri)?.use { input ->
            cached.outputStream().use { output -> input.copyTo(output) }
          } ?: throw IllegalArgumentException("Не удалось открыть аудиопоток")
          tempCopy = cached
          extractor.setDataSource(cached.absolutePath)
        }
      }
      val trackIndex = (0 until extractor.trackCount).firstOrNull {
        extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
      } ?: throw IllegalArgumentException("В файле не найдена звуковая дорожка")
      extractor.selectTrack(trackIndex)
      val format = extractor.getTrackFormat(trackIndex)
      val mime = format.getString(MediaFormat.KEY_MIME)!!
      val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
      var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
      var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

      val codec = MediaCodec.createDecoderByType(mime)
      codec.configure(format, null, null, 0)
      codec.start()

      val expectedOut16k = if (durationUs > 0) {
        ((durationUs / 1_000_000.0) * TARGET_RATE).toInt() + TARGET_RATE
      } else {
        1 shl 20
      }
      val out = FloatArrayBuilder(expectedOut16k)
      var resampler = StreamingResampler(sampleRate, TARGET_RATE, out)
      var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
      val info = MediaCodec.BufferInfo()
      var inputDone = false
      var outputDone = false
      while (!outputDone) {
        if (!inputDone) {
          val inIndex = codec.dequeueInputBuffer(10_000)
          if (inIndex >= 0) {
            val buffer = codec.getInputBuffer(inIndex)!!
            val size = extractor.readSampleData(buffer, 0)
            if (size < 0) {
              codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
              inputDone = true
            } else {
              val pts = extractor.sampleTime
              codec.queueInputBuffer(inIndex, 0, size, pts, 0)
              extractor.advance()
              if (durationUs > 0) onProgress((pts.toFloat() / durationUs).coerceIn(0f, 0.99f))
            }
          }
        }
        val outIndex = codec.dequeueOutputBuffer(info, 10_000)
        when {
          outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
            val f = codec.outputFormat
            val newRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) pcmEncoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
            if (newRate != sampleRate) {
              sampleRate = newRate
              resampler = StreamingResampler(sampleRate, TARGET_RATE, out)
            }
          }
          outIndex >= 0 -> {
            val buffer = codec.getOutputBuffer(outIndex)!!
            if (info.size > 0) {
              buffer.position(info.offset)
              buffer.limit(info.offset + info.size)
              appendInterleavedResampled(
                buf = buffer.slice().order(ByteOrder.nativeOrder()),
                encoding = pcmEncoding,
                channels = channels,
                resampler = resampler,
              )
            }
            codec.releaseOutputBuffer(outIndex, false)
            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
          }
        }
      }
      resampler.finish()
      codec.stop()
      codec.release()
      return PcmAudio(out.toExactArray(), TARGET_RATE)
    } finally {
      runCatching { extractor.release() }
      tempCopy?.delete()
    }
  }

  /** Сводит многоканальный сигнал в моно и сразу ресемплирует в 16 кГц на лету. */
  private fun appendInterleavedResampled(
    buf: ByteBuffer,
    encoding: Int,
    channels: Int,
    resampler: StreamingResampler,
  ) {
    when (encoding) {
      AudioFormat.ENCODING_PCM_FLOAT -> {
        val fb = buf.asFloatBuffer()
        val frames = fb.remaining() / channels
        for (i in 0 until frames) {
          var acc = 0f
          for (c in 0 until channels) acc += fb.get()
          resampler.accept(acc / channels)
        }
      }
      else -> {
        val sb = buf.asShortBuffer()
        val frames = sb.remaining() / channels
        for (i in 0 until frames) {
          var acc = 0f
          for (c in 0 until channels) acc += sb.get() / 32768f
          resampler.accept(acc / channels)
        }
      }
    }
  }
}

/**
 * Растущий массив float без упаковки в объекты.
 * При расширении на больших объёмах (> 8 МБ) растёт шагами по 25% вместо удвоения,
 * чтобы не вызывать OOM на длинных аудиозаписях.
 */
class FloatArrayBuilder(initial: Int = 1 shl 16) {
  private var data = FloatArray(maxOf(initial, 16))
  var size = 0
    private set

  private fun ensureCapacity(minCap: Int) {
    if (minCap <= data.size) return
    val grow = if (data.size >= 2_000_000) {
      data.size + (data.size shr 2)
    } else {
      data.size * 2
    }
    data = data.copyOf(maxOf(grow, minCap))
  }

  fun add(v: Float) {
    if (size == data.size) ensureCapacity(size + 1)
    data[size++] = v
  }

  fun addAll(src: FloatArray, from: Int = 0, to: Int = src.size) {
    val n = to - from
    ensureCapacity(size + n)
    System.arraycopy(src, from, data, size, n)
    size += n
  }

  fun toArray(): FloatArray = data.copyOf(size)

  /**
   * Возвращает внутренний массив без лишнего копирования, если буфер заполнен почти полностью,
   * экономя сотни мегабайт кучи на 45–60-минутных записях.
   */
  fun toExactArray(): FloatArray {
    return if (size == data.size) {
      data
    } else {
      val trimmed = data.copyOf(size)
      data = FloatArray(0)
      trimmed
    }
  }
}
