all: doomcode.bin doomcode.h

doomcode.h: doomcode.bin
	python mkhdr.py doomcode.bin doomcode.h

doomcode.bin: doomcode.sasm
	python doomasm.py doomcode.sasm doomcode.bin
