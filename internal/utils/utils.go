package utils

import (
	"math"
	"unicode"
	"unicode/utf8"
)

func EstimateTokens(text string) int64 {
	if text == "" {
		return 0
	}

	var asciiCount int
	var cjkCount int
	var otherBytes int

	for _, r := range text {
		switch {
		case r <= unicode.MaxASCII:
			// 英文、数字、空格和 ASCII 标点
			asciiCount++

		case unicode.In(
			r,
			unicode.Han,
			unicode.Hiragana,
			unicode.Katakana,
			unicode.Hangul,
		):
			// 中文、日文、韩文
			cjkCount++

		default:
			// Emoji、特殊符号和其他 Unicode 字符
			size := max(utf8.RuneLen(r), 1)
			otherBytes += size
		}
	}

	estimate :=
		float64(asciiCount)/4.0 +
			float64(cjkCount) +
			float64(otherBytes)/2.0

	return int64(math.Ceil(estimate))
}
