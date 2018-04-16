#!/usr/bin/env python3

from argparse import ArgumentParser
from binflakes.types import BinArray

parser = ArgumentParser(description="Converts the microcode from raw binary to C header file.")
parser.add_argument('input', help="the input file")
parser.add_argument('output', help="the output file")

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        d = f.read()

    d = BinArray(d)
    d = d.repack(32, msb_first=False)

    with open(args.output, 'w') as f:
        f.write('static const uint32_t doomcode[] = {\n')
        for x in d:
            f.write(f'\t0x{x.to_uint():08x},\n')
        f.write('};\n')
