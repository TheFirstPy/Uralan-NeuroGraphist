package ru.hexteam.stenograf

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import ru.hexteam.stenograf.audio.FloatArrayBuilder
import ru.hexteam.stenograf.audio.Resampler
import ru.hexteam.stenograf.audio.StreamingResampler
import ru.hexteam.stenograf.audio.VoicePitchAnalyzer
import ru.hexteam.stenograf.audio.WavParser
import ru.hexteam.stenograf.audio.WavWriter
import ru.hexteam.stenograf.core.export.TranscriptExporter
import ru.hexteam.stenograf.core.model.Utterance
import ru.hexteam.stenograf.core.pipeline.DiarizationSlice
import ru.hexteam.stenograf.core.pipeline.RawToken
import ru.hexteam.stenograf.core.pipeline.TranscriptAligner
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class TranscriptPipelineTest {

  @Test
  fun `JSON-экспорт строго соответствует схеме хакатона (стр 15)`() {
    val utterances = listOf(
      Utterance(
        id = 1,
        speakerId = "speaker_1",
        startMs = 1250L,
        endMs = 4380L,
        text = "Добрый день!",
      ),
      Utterance(
        id = 2,
        speakerId = "speaker_2",
        startMs = 4900L,
        endMs = 8120L,
        text = "Здравствуйте, начинаем совещание.",
      ),
    )

    val jsonStr = TranscriptExporter.toHackathonJson(utterances)
    val root = Json.parseToJsonElement(jsonStr).jsonObject
    assertTrue("Корневой ключ transcript обязателен", root.containsKey("transcript"))
    val arr = root["transcript"]!!.jsonArray
    assertEquals(2, arr.size)

    val first = arr[0].jsonObject
    assertEquals("speaker_1", first["speaker_id"]!!.jsonPrimitive.content)
    assertEquals(1250L, first["start_ms"]!!.jsonPrimitive.content.toLong())
    assertEquals(4380L, first["end_ms"]!!.jsonPrimitive.content.toLong())
    assertEquals("Добрый день!", first["text"]!!.jsonPrimitive.content)
  }

  @Test
  fun `Выравнивание BPE-токенов и сегментов диаризации разделяет реплики двух говорящих`() {
    val tokens = listOf(
      RawToken("▁Алло", 100L),
      RawToken(",", 350L),
      RawToken("▁здравствуйте", 500L),
      RawToken("!", 950L),
      RawToken("▁Добрый", 1800L),
      RawToken("▁день", 2200L),
      RawToken(".", 2550L),
    )
    val words = TranscriptAligner.tokensToWords(tokens, chunkStartMs = 1000L, chunkEndMs = 4000L)
    assertEquals(4, words.size)
    assertEquals("Алло,", words[0].text)
    assertEquals("здравствуйте!", words[1].text)
    assertEquals("Добрый", words[2].text)
    assertEquals("день.", words[3].text)

    val slices = listOf(
      DiarizationSlice(startMs = 900L, endMs = 2300L, speakerIndex = 7),
      DiarizationSlice(startMs = 2600L, endMs = 4000L, speakerIndex = 3),
    )
    val assigned = TranscriptAligner.assignSpeakersToWords(words, slices)
    val utterances = TranscriptAligner.buildUtterances(assigned)

    assertEquals(2, utterances.size)
    assertEquals("speaker_1", utterances[0].speakerId)
    assertEquals("Алло, здравствуйте!", utterances[0].text)
    assertEquals("speaker_2", utterances[1].speakerId)
    assertEquals("Добрый день.", utterances[1].text)
    assertTrue(utterances[0].startMs < utterances[0].endMs)
    assertTrue(utterances[1].startMs >= utterances[0].endMs)
  }

  @Test
  fun `Потоковый StreamingResampler побитово совпадает с оконным sinc-фильтром Блэкмана Resampler`() {
    val src48k = FloatArray(48_000) { i ->
      (sin(i * 0.04) * 0.4 + sin(i * 0.21) * 0.3).toFloat()
    }
    val expected = Resampler.resample(src48k, 48_000, 16_000)
    val outBuilder = FloatArrayBuilder(expected.size + 16)
    val streaming = StreamingResampler(48_000, 16_000, outBuilder)
    for (s in src48k) streaming.accept(s)
    streaming.finish()
    val actual = outBuilder.toExactArray()

    assertEquals(expected.size, actual.size)
    var maxDiff = 0f
    for (i in expected.indices) {
      val d = abs(expected[i] - actual[i])
      if (d > maxDiff) maxDiff = d
    }
    assertTrue("Максимальное расхождение sinc-ресемплера = $maxDiff", maxDiff < 1e-5f)

    val tmp = File.createTempFile("test_48k_", ".wav")
    try {
      WavWriter.write(tmp, src48k, 48_000)
      val parsed = tmp.inputStream().use { WavParser.parse(it, targetSampleRate = 16_000) }
      assertNotNull(parsed)
      assertEquals(16_000, parsed!!.sampleRate)
      assertEquals(16_000, parsed.samples.size)
    } finally {
      tmp.delete()
    }
  }

  @Test
  fun `Парсинг кликабельных таймкодов из ИИ-протокола для перехода к аудиофрагменту`() {
    assertEquals(84_500L, TranscriptExporter.extractFirstTimestampMs("`[01:24.5]` **Говорящий 1**: Обсудили архитектуру."))
    assertEquals(12_900L, TranscriptExporter.extractFirstTimestampMs("[00:12.9] Говорящий 2: Принято."))
  }

  @Test
  fun `Диалоговое тире GigaAM v3 разделяет фразы и не приклеивается к предыдущему слову`() {
    val tokens = listOf(
      RawToken("▁Э-э-э", 50L),
      RawToken("...", 90L),
      RawToken("▁Ведёт", 120L),
      RawToken("▁и", 350L),
      RawToken("▁тащит", 550L),
      RawToken("▁разные", 800L),
      RawToken("▁вещи", 1100L),
      RawToken(".", 1300L),
      RawToken("▁—", 1360L),
      RawToken("▁Ну", 1420L),
      RawToken(",", 1520L),
      RawToken("▁представь", 1700L),
      RawToken("▁уснула", 2200L),
      RawToken(".", 2500L),
    )
    val words = TranscriptAligner.tokensToWords(tokens, chunkStartMs = 0L, chunkEndMs = 3000L)
    assertEquals(8, words.size)
    assertEquals("Ведёт", words[0].text)
    assertEquals("вещи.", words[4].text)
    assertTrue("Слово после тире должно иметь turnBreakBefore=true", words[5].turnBreakBefore)
    assertEquals("Ну,", words[5].text)

    val phrases = TranscriptAligner.buildPhraseSegments(words)
    assertEquals(2, phrases.size)
    assertEquals("Ведёт и тащит разные вещи.", phrases[0].text)
    assertEquals("Ну, представь уснула.", phrases[1].text)
    assertTrue(phrases[1].hasTurnDash)
  }

  @Test
  fun `VoicePitchAnalyzer точно различает мужской (100 Гц) и женский (200 Гц) основной тон`() {
    val sr = 16_000
    val maleWave = FloatArray(sr) { i ->
      val t = i.toDouble() / sr
      (0.6 * sin(2 * PI * 100.0 * t) + 0.3 * sin(2 * PI * 200.0 * t)).toFloat()
    }
    val femaleWave = FloatArray(sr) { i ->
      val t = i.toDouble() / sr
      (0.6 * sin(2 * PI * 200.0 * t) + 0.3 * sin(2 * PI * 400.0 * t)).toFloat()
    }
    val f0Male = VoicePitchAnalyzer.computeMedianF0Hz(maleWave, sampleRate = sr)
    val f0Female = VoicePitchAnalyzer.computeMedianF0Hz(femaleWave, sampleRate = sr)
    assertTrue("Мужской F0 ($f0Male Гц) должен быть около 100 Гц", f0Male in 94f..106f)
    assertTrue("Женский F0 ($f0Female Гц) должен быть около 200 Гц", f0Female in 190f..212f)
    assertTrue(VoicePitchAnalyzer.f0Distance(f0Male, f0Female) > 0.14f)
  }
}
