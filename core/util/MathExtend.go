package util

// MaxValue : 指定した配列の中で最大値となる値を取得
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

// ApplyValues : 指定した処理を配列の各値に実行する
func ApplyValues(vs []float64, fn func(key int, value float64) float64) []float64 {
	results := make([]float64, 0, len(vs))
	for key, v := range vs {
		results = append(results, fn(key, v))
	}
	return results
}

// AddArray : 配列同士を加算する
func AddArray(vs1 []float64, vs2 []float64) []float64 {
	if len(vs1) != len(vs2) {
		panic("AddArray : 指定した配列の要素数に差分があります")
	}

	return ApplyValues(vs1, func(key int, value float64) float64 {
		return value + vs2[key]
	})
}

// ScaleArray : 配列の各要素に対して指定倍率をかける
func ScaleArray(vs []float64, scale float64) []float64 {
	return ApplyValues(vs, func(key int, value float64) float64 {
		return value * scale
	})
}

// SumValues : 配列の要素を合計する
func SumValues(vs []float64) float64 {
	result := 0.0
	for _, v := range vs {
		result += v
	}
	return result
}
