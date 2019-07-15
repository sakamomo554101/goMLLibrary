.PHONY: clean all

SOURCES=$(wildcard *.go)
EXECUTABLE=$(patsubst %.go, %, $(SOURCES))
LIB_PATH=${GOPATH}/src

all: $(EXECUTABLE)
	@golint

%: %.o
	@go tool link -linkmode external -extld "g++" -extldflags "-ldl" -o $@ $<

%.o: %.go
	@go tool compile -pack -o $@ $<

clean:
	@rm -f $(EXECUTABLE) *.so *.o *.a
