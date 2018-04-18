package util

func MaxValue(vs []float64) (key int, max float64) {
	max = vs[0]
	key = 0
	for k, v := range vs {
		if max < v {
			max = v
			key = k
		}
	}
	return key, max
}
