package ru.hexteam.stenograf.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Запись речи с микрофона устройства в WAV 16 кГц моно PCM16.
 * Отдаёт текущую длительность и RMS-амплитуду для отрисовки живой звуковой волны.
 */
class AudioRecorder(private val scope: CoroutineScope) {

  private var job: Job? = null
  private var outputFile: File? = null

  private val _isRecording = MutableStateFlow(false)
  val isRecording: StateFlow<Boolean> = _isRecording

  private val _elapsedMs = MutableStateFlow(0L)
  val elapsedMs: StateFlow<Long> = _elapsedMs

  private val _amplitude = MutableStateFlow(0f)
  val amplitude: StateFlow<Float> = _amplitude

  @SuppressLint("MissingPermission")
  fun start(targetFile: File) {
    if (_isRecording.value) return
    val sampleRate = AudioDecoder.TARGET_RATE
    val minBuf = AudioRecord.getMinBufferSize(
      sampleRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
    ).coerceAtLeast(sampleRate / 5 * 2)

    val record = AudioRecord(
      MediaRecorder.AudioSource.MIC,
      sampleRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
      minBuf * 2,
    )
    require(record.state == AudioRecord.STATE_INITIALIZED) {
      "Не удалось инициализировать микрофон"
    }

    targetFile.parentFile?.mkdirs()
    outputFile = targetFile
    _elapsedMs.value = 0L
    _amplitude.value = 0f
    _isRecording.value = true

    job = scope.launch(Dispatchers.IO) {
      val shortBuf = ShortArray(sampleRate / 10) // окно 100 мс
      val byteBuf = ByteBuffer.allocate(shortBuf.size * 2).order(ByteOrder.LITTLE_ENDIAN)
      var totalSamples = 0L

      FileOutputStream(targetFile).use { fos ->
        fos.write(WavWriter.header(0, sampleRate))
        record.startRecording()
        try {
          while (isActive && _isRecording.value) {
            val read = record.read(shortBuf, 0, shortBuf.size)
            if (read > 0) {
              byteBuf.clear()
              var sumSq = 0.0
              for (i in 0 until read) {
                val s = shortBuf[i]
                byteBuf.putShort(s)
                val norm = s / 32768.0
                sumSq += norm * norm
              }
              fos.write(byteBuf.array(), 0, read * 2)
              totalSamples += read
              _elapsedMs.value = totalSamples * 1000L / sampleRate
              _amplitude.value = sqrt(sumSq / read).toFloat().coerceIn(0f, 1f)
            }
          }
        } finally {
          try {
            record.stop()
          } catch (_: Exception) {
          }
          record.release()
        }
      }
      WavWriter.patchHeader(targetFile, sampleRate)
    }
  }

  suspend fun stop(): File? {
    if (!_isRecording.value) return null
    _isRecording.value = false
    job?.join()
    job = null
    _amplitude.value = 0f
    return outputFile
  }
}
