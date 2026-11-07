package ru.hexteam.stenograf.audio

import android.media.MediaPlayer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

/**
 * Контроллер воспроизведения аудиозаписи с переходом к выбранной реплике или слову
 * и отслеживанием текущей позиции в миллисекундах для подсветки активной реплики в диалоге.
 */
class AudioPlayerController(private val scope: CoroutineScope) {

  private var mediaPlayer: MediaPlayer? = null
  private var currentPath: String? = null
  private var tickerJob: Job? = null
  private var stopAtMs: Long? = null

  private val _isPlaying = MutableStateFlow(false)
  val isPlaying: StateFlow<Boolean> = _isPlaying

  private val _positionMs = MutableStateFlow(0L)
  val positionMs: StateFlow<Long> = _positionMs

  private val _durationMs = MutableStateFlow(0L)
  val durationMs: StateFlow<Long> = _durationMs

  fun load(audioPath: String) {
    if (currentPath == audioPath && mediaPlayer != null) return
    release()
    val file = File(audioPath)
    if (!file.exists()) return
    runCatching {
      val mp = MediaPlayer().apply {
        setDataSource(file.absolutePath)
        prepare()
        setOnCompletionListener {
          _isPlaying.value = false
          _positionMs.value = 0L
          stopAtMs = null
          tickerJob?.cancel()
        }
      }
      mediaPlayer = mp
      currentPath = audioPath
      _durationMs.value = mp.duration.toLong().coerceAtLeast(0L)
      _positionMs.value = 0L
    }
  }

  fun togglePlayPause(audioPath: String) {
    load(audioPath)
    val mp = mediaPlayer ?: return
    if (mp.isPlaying) {
      mp.pause()
      _isPlaying.value = false
      tickerJob?.cancel()
    } else {
      stopAtMs = null
      mp.start()
      _isPlaying.value = true
      startTicker()
    }
  }

  /** Переход к реплике [startMs] и воспроизведение. Если задан [endMs], можно остановить по окончании. */
  fun playFrom(audioPath: String, startMs: Long, endMs: Long? = null) {
    load(audioPath)
    val mp = mediaPlayer ?: return
    stopAtMs = endMs
    mp.seekTo(startMs.toInt().coerceAtLeast(0))
    _positionMs.value = startMs.coerceAtLeast(0L)
    if (!mp.isPlaying) {
      mp.start()
    }
    _isPlaying.value = true
    startTicker()
  }

  fun seekTo(ms: Long) {
    val mp = mediaPlayer ?: return
    stopAtMs = null
    val target = ms.coerceIn(0L, _durationMs.value)
    mp.seekTo(target.toInt())
    _positionMs.value = target
  }

  fun pause() {
    mediaPlayer?.let {
      if (it.isPlaying) it.pause()
    }
    _isPlaying.value = false
    tickerJob?.cancel()
  }

  private fun startTicker() {
    tickerJob?.cancel()
    tickerJob = scope.launch(Dispatchers.Main) {
      while (isActive && _isPlaying.value) {
        val mp = mediaPlayer ?: break
        if (!mp.isPlaying) {
          _isPlaying.value = false
          break
        }
        val pos = mp.currentPosition.toLong()
        _positionMs.value = pos
        val limit = stopAtMs
        if (limit != null && pos >= limit) {
          mp.pause()
          _isPlaying.value = false
          stopAtMs = null
          break
        }
        delay(50L)
      }
    }
  }

  fun release() {
    tickerJob?.cancel()
    runCatching {
      mediaPlayer?.stop()
    }
    runCatching {
      mediaPlayer?.release()
    }
    mediaPlayer = null
    currentPath = null
    _isPlaying.value = false
    _positionMs.value = 0L
  }
}
