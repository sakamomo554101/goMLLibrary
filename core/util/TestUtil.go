package util

// CreateFloatArrayByStep : floatの配列を作成
func CreateFloatArrayByStep(count int, firstValue float64, stepValue float64) []float64 {
	array := make([]float64, 0, count)
	for i := 0; i < count; i++ {
		val := firstValue + stepValue*float64(i)
		array = append(array, val)
	}
	return array
}

// CreateSameValueArray : floatの配列を作成（すべて同じ値）
func CreateSameValueArray(count int, value float64) []float64 {
	array := make([]float64, 0, count)
	for i := 0; i < count; i++ {
		array = append(array, value)
	}
	return array
}
