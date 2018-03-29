## Makefile
VERSION := 0.0.1

.PHONY: deps
deps:
	glide install -v

.PHONY: update
update:
	glide up
