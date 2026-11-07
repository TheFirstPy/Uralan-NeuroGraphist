package ru.hexteam.stenograf.ui.theme

import androidx.compose.ui.graphics.Color

// Фирменная палитра КСиТ МИФИ × HEX.TEAM (Nuclear IT Hack)
val primaryLight = Color(0xFF0B46C4)
val primaryContainerLight = Color(0xFFDCE4FF)
val secondaryLight = Color(0xFF006879)
val tertiaryLight = Color(0xFF6B3DD1)

val primaryDark = Color(0xFFB3C5FF)
val primaryContainerDark = Color(0xFF00319C)
val secondaryDark = Color(0xFF63D3EA)
val tertiaryDark = Color(0xFFD1BCFF)

/** Индивидуальные цвета карточек и бейджей для разных говорящих. */
val SpeakerColors = listOf(
  Color(0xFF0B46C4), // speaker_1 — кобальтовый синий
  Color(0xFF00796B), // speaker_2 — изумрудно-бирюзовый
  Color(0xFFAD1457), // speaker_3 — рубиновый
  Color(0xFFE65100), // speaker_4 — янтарный
  Color(0xFF5E35B1), // speaker_5 — индиго
  Color(0xFF2E7D32), // speaker_6 — хвойный
)

fun speakerAccentColor(speakerId: String): Color {
  val idx = (speakerId.substringAfterLast('_').toIntOrNull() ?: 1) - 1
  return SpeakerColors[idx.mod(SpeakerColors.size)]
}
