## Makefile
VERSION := 0.0.1

.PHONY: deps
deps:
	# depツールのインストール
	go get -u github.com/golang/dep/cmd/dep
	# vendor配下に依存パッケージをインストール
	dep ensure
	# GoMNISTの取得(TODO : なぜかGoMNISTのバイナリデータがdep ensureでは入らないため、Go Getしている)
	go get -u github.com/petar/GoMNIST
	rm -rf $(shell pwd)/vendor/github.com/petar/

.PHONY: update
update:
	# TODO : imp
