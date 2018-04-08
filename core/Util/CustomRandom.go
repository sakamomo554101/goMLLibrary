package Util

import (
	"math/rand"
	"time"
)

func randomFloat(r *rand.Rand, min, max float64) float64 {
	return r.Float64()*(max-min) + min
}

func RandomFloatArray(min, max float64, count int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	list := make([]float64, 0, count)
	for i := 0; i < count; i++ {
		v := randomFloat(r, min, max)
		list = append(list, v)
	}
	return list
}
