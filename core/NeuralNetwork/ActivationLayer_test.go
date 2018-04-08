package NeuralNetwork

import "testing"

func TestHoge(t *testing.T) {
	s := NewSigmoid()
	s.Forward(nil)
	t.Errorf("hoge")
}
