all: test info foo

test: test.c
	gcc -o test test.c -framework OpenCL

info: info.c
	gcc -o info info.c -framework OpenCL

foo: foo.c
	gcc -o foo foo.c -framework OpenCL

clean:
	rm -f test info foo

.PHONY: clean
