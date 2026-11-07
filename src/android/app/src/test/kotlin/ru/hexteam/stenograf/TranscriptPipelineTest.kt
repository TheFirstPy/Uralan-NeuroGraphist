package ru.hexteam.stenograf

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import ru.hexteam.stenograf.audio.WavParser
import ru.hexteam.stenograf.audio.WavWriter
import ru.hexteam.stenograf.core.export.TranscriptExporter
import ru.hexteam.stenograf.core.model.Utterance
import ru.hexteam.stenograf.core.model.Word
import ru.hexteam.stenograf.core.pipeline.DiarizationSlice
import ru.hexteam.stenograf.core.pipeline.RawToken
import ru.hexteam.stenograf.core.pipeline.TranscriptAligner
import java.io.ByteArrayInputStream
import java.io.File

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
  fun `Потоковый WavParser ресемплирует 48 кГц в 16 кГц на лету без лишних буферов`() {
    val tmp = File.createTempFile("test_48k_", ".wav")
    try {
      // 2 секунды сигнала на частоте 48 000 Гц (96 000 отсчётов)
      val src48k = FloatArray(96_000) { i -> kotlin.math.sin(i * 0.05).toFloat() * 0.5f }
      WavWriter.write(tmp, src48k, 48_000)
      val parsed = tmp.inputStream().use { WavParser.parse(it, targetSampleRate = 16_000) }
      assertNotNull(parsed)
      assertEquals(16_000, parsed!!.sampleRate)
      // Ожидаем ~32 000 отсчётов (2 секунды * 16 000 Гц)
      assertTrue("Размер=${parsed.samples.size}", parsed.samples.size in 31_990..32_010)
    } finally {
      tmp.delete()
    }
  }

  @Test
  fun `Фильтрация шумовых одиночных кластеров сохраняет реальных 3 спикеров и убирает 4-й шумовой выброс`() {
    // 8 реплик: спикеры 0, 1 и 2 ведут диалог, а одна микро-реплика (300 мс) ошибочно попала в кластер 99
    val assigned = listOf(
      Word("Привет", 0L, 4000L) to 0,
      Word("Здравствуйте", 4500L, 9000L) to 1,
      Word("Добрый", 9500L, 14000L) to 2,
      Word("Начинаем", 14500L, 19000L) to 0,
      Word("Отлично", 19500L, 24000L) to 1,
      Word("Согласен", 24500L, 29000L) to 2,
      Word("Эх", 29200L, 29500L) to 99, // Одиночный шумовой выброс 300 мс (4-й кластер)
      Word("Продолжаем", 30000L, 35000L) to 1,
    )
    val utterances = TranscriptAligner.buildUtterances(assigned, filterMinorSpeakers = true)
    val distinctSpeakers = utterances.map { it.speakerId }.distinct()
    assertEquals(listOf("speaker_1", "speaker_2", "speaker_3"), distinctSpeakers)
  }

  @Test
  fun `Парсинг кликабельных таймкодов из ИИ-протокола и детекция перебиваний`() {
    assertEquals(84_500L, TranscriptExporter.extractFirstTimestampMs("`[01:24.5]` **Говорящий 1**: Обсудили архитектуру."))
    assertEquals(12_900L, TranscriptExporter.extractFirstTimestampMs("[00:12.9] Говорящий 2: Принято."))

    val sr = 16_000
    val samples = FloatArray(sr * 6) { i ->
      // Первые 3 секунды — обычная громкость (0.1), вторые 3 секунды — повышенный тон (0.45)
      val amp = if (i < sr * 3) 0.10f else 0.45f
      kotlin.math.sin(i * 0.1).toFloat() * amp
    }
    val rawUtterances = listOf(
      Utterance(id = 1, speakerId = "speaker_1", startMs = 0L, endMs = 2800L, text = "Спокойная реплика."),
      Utterance(id = 2, speakerId = "speaker_2", startMs = 2850L, endMs = 5800L, text = "Срочно исправляем ошибку!"),
    )
    val slices = listOf(
      DiarizationSlice(0L, 3100L, 0),
      DiarizationSlice(2700L, 5800L, 1), // перекрытие 400 мс (2700..3100)
    )
    val enriched = TranscriptAligner.enrichProsodyAndOverlaps(rawUtterances, slices, samples, sr)
    assertTrue("Вторая реплика должна иметь флаг перебивания/наложения", enriched[1].hasOverlap)
    assertEquals("intense", enriched[1].prosodyTag)
  }
}
