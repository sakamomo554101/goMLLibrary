package util

import (
	"math/rand"
	"time"
)

func randomFloat(r *rand.Rand, min, max float64) float64 {
	return r.Float64()*(max-min) + min
}

func randomFloat32(r *rand.Rand, min, max float32) float32 {
	return r.Float32()*(max-min) + min
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

func RandomFloatArray32(min, max float32, count int) []float32 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	list := make([]float32, 0, count)
	for i := 0; i < count; i++ {
		v := randomFloat32(r, min, max)
		list = append(list, v)
	}
	return list
}

func RandomIntArray(max int, count int) []int {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	list := make([]int, 0, count)
	for i := 0; i < count; i++ {
		list = append(list, r.Intn(max))
	}
	return list
}

func NormRandomArray(stdenv float64, count int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	list := make([]float64, count)
	for i, _ := range list {
		// NormFloat64は平均0, 標準偏差1の正規分布を作成
		list[i] = r.NormFloat64() * stdenv
	}
	return list
}
