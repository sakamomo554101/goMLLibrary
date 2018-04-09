package NeuralNetwork

func createFloatArrayByStep(count int, firstValue float64, stepValue float64) []float64 {
	array := make([]float64, 0, count)
	for i := 0; i < count; i++ {
		val := firstValue + stepValue*float64(i)
		array = append(array, val)
	}
	return array
}
