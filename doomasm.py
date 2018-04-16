#!/usr/bin/env python3

"""A dead simple microcode assembler for HardDoom™ microcode.

Based on binflakes S-expression syntax, because I can't be bothered to write
a proper assembly parser.

Accepts 4 things in the source:

- (register abc 12): defines the name abc as register #12
- (const abc 12): defines the name abc as the constant 12
- (label abc): defines the name abc as a constant equal to the current code
  position
- (<instruction> <arguments>...): arguments can be immediate constants, names
  defined as constants or registers, or simple constant expressions involving +.

Works in two simple passes:

- first pass: determines the value of every name, counts instructions to know
  the current position for labels
- second pass: assembles the instructions
"""

from argparse import ArgumentParser

from attr import attrs, attrib
from binflakes.sexpr.symbol import Symbol
from binflakes.sexpr.read import read_file
from binflakes.sexpr.nodes import (
        FormNode, SymbolNode, IntNode, AlternativesNode, form_node, form_arg,
)
from sys import exit

parser = ArgumentParser(description="Assemble HardDoom™ microcode.")
parser.add_argument('source', help="the source file")
parser.add_argument('output', help="the output file")


MAX_INSNS = 0x1000


@attrs
class Register:
    idx = attrib()


@form_node
class RegisterNode(FormNode):
    symbol = Symbol('register')
    name = form_arg(SymbolNode)
    index = form_arg(IntNode)

    def first(self, ctx):
        if self.name.value in ctx.names:
            print(f'{self.location}: redefined name {self.name}')
            exit(1)
        ctx.names[self.name.value] = Register(self.index.value)


@form_node
class ConstNode(FormNode):
    symbol = Symbol('const')
    name = form_arg(SymbolNode)
    value = form_arg(IntNode)

    def first(self, ctx):
        if self.name.value in ctx.names:
            print(f'{self.location}: redefined name {self.name.value}')
            exit(1)
        ctx.names[self.name.value] = self.value.value


@form_node
class LabelNode(FormNode):
    symbol = Symbol('label')
    name = form_arg(SymbolNode)

    def first(self, ctx):
        if self.name.value in ctx.names:
            print(f'{self.location}: redefined name {self.name.value}')
            exit(1)
        ctx.names[self.name.value] = ctx.pos


class InsnNode:
    def first(self, ctx):
        if ctx.pos >= MAX_INSNS:
            print(f'{self.location}: code length limit reached')
            exit(1)
        ctx.pos += 1


class ImmNode(AlternativesNode):
    pass


class BaseImmNode:
    def eval(self, ctx, bits, offset=0, signed=False):
        res = self.calc(ctx)
        res -= offset
        if signed:
            sign = 1 << (bits - 1)
            res += sign
            res ^= sign
        mask = (1 << bits) - 1
        if res & ~mask:
            print(f'{self.location}: value {res} does not fit in {bits} bits')
            exit(1)
        return res


class ImmConstNode(IntNode, BaseImmNode):
    def calc(self, ctx):
        return self.value


class ImmNameNode(SymbolNode, BaseImmNode):
    def calc(self, ctx):
        if self.value not in ctx.names:
            print(f'{self.location}: undefined name {self.value}')
            exit(1)
        res = ctx.names[self.value]
        if isinstance(res, Register):
            print(f'{self.location}: immediate expected, got register')
            exit(1)
        return res


@form_node
class ImmPlusNode(FormNode, BaseImmNode):
    symbol = Symbol('+')
    el = form_arg(ImmNode)
    er = form_arg(ImmNode)

    def calc(self, ctx):
        return self.el.calc(ctx) + self.er.calc(ctx)


@form_node
class ImmMinusNode(FormNode, BaseImmNode):
    symbol = Symbol('-')
    el = form_arg(ImmNode)
    er = form_arg(ImmNode)

    def calc(self, ctx):
        return self.el.calc(ctx) - self.er.calc(ctx)


ImmNode.set_alternatives([
    ImmNameNode,
    ImmConstNode,
    ImmPlusNode,
    ImmMinusNode,
])


class RegNode(SymbolNode):
    def eval(self, ctx):
        if self.value not in ctx.names:
            print(f'{self.location}: undefined name {self.value}')
            exit(1)
        res = ctx.names[self.value]
        if not isinstance(res, Register):
            print(f'{self.location}: register expected, got immediate')
            exit(1)
        return res.idx


def asm_rri12(op1, op2, r1, r2, imm):
    assert not (op1 & ~0x3f)
    assert not (op2 & ~0xf)
    assert not (r1 & ~0x1f)
    assert not (r2 & ~0x1f)
    assert not (imm & ~0xfff)
    return op1 << 26 | op2 << 22 | r1 << 17 | r2 << 12 | imm

def asm_ri7i12(op1, op2, r1, imm1, imm2):
    assert not (op1 & ~0x3f)
    assert not (op2 & ~0x3)
    assert not (r1 & ~0x1f)
    assert not (imm1 & ~0x7f)
    assert not (imm2 & ~0xfff)
    return op1 << 26 | op2 << 24 | r1 << 19 | imm1 << 12 | imm2

def asm_rrrrr(op1, op2, r1, r2, r3, r4, r5):
    assert not (op1 & ~0x3f)
    assert not (op2 & ~0x1)
    assert not (r1 & ~0x1f)
    assert not (r2 & ~0x1f)
    assert not (r3 & ~0x1f)
    assert not (r4 & ~0x1f)
    assert not (r5 & ~0x1f)
    return op1 << 26 | op2 << 25 | r1 << 20 | r2 << 15 | r3 << 10 | r4 << 5 | r5

def asm_rrr(op1, op2, r1, r2, r3):
    assert not (op1 & ~0x3f)
    assert not (op2 & ~0x7ff)
    assert not (r1 & ~0x1f)
    assert not (r2 & ~0x1f)
    assert not (r3 & ~0x1f)
    return op1 << 26 | op2 << 15 | r1 << 10 | r2 << 5 | r3

def asm_rri11r(op1, r1, r2, imm, r3):
    assert not (op1 & ~0x3f)
    assert not (r1 & ~0x1f)
    assert not (r2 & ~0x1f)
    assert not (imm & ~0x7ff)
    assert not (r3 & ~0x1f)
    return op1 << 26 | r1 << 21 | r2 << 16 | imm << 5 | r3

def asm_rri16(op1, r1, r2, imm):
    assert not (op1 & ~0x3f)
    assert not (r1 & ~0x1f)
    assert not (r2 & ~0x1f)
    assert not (imm & ~0xffff)
    return op1 << 26 | r1 << 21 | r2 << 16 | imm


def form_ozz12(op1, op2, op3, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, op3, 0, 0)

    return MyNode


def form_orz12(op1, op2, op3, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg = form_arg(RegNode)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, op3, self.reg.eval(ctx), 0)

    return MyNode


def form_ozi12(op1, op2, op3, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        imm = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, op3, 0, self.imm.eval(ctx, 12))

    return MyNode


def form_ori12(op1, op2, op3, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg = form_arg(RegNode)
        imm = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, op3, self.reg.eval(ctx), self.imm.eval(ctx, 12))

    return MyNode


def form_osri12(op1, op2, op3, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        imm = form_arg(ImmNode)
        reg = form_arg(RegNode)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, op3, self.reg.eval(ctx), self.imm.eval(ctx, 12))

    return MyNode


def form_rii12(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg = form_arg(RegNode)
        imm1 = form_arg(ImmNode)
        imm2 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, self.reg.eval(ctx), self.imm1.eval(ctx, 5), self.imm2.eval(ctx, 12))

    return MyNode


def form_rri12(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)
        imm = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri12(op1, op2, self.reg1.eval(ctx), self.reg2.eval(ctx), self.imm.eval(ctx, 12))

    return MyNode


def form_ri7i12(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg = form_arg(RegNode)
        imm1 = form_arg(ImmNode)
        imm2 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_ri7i12(op1, op2, self.reg.eval(ctx), self.imm1.eval(ctx, 7), self.imm2.eval(ctx, 12))

    return MyNode


def form_rzi16(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg = form_arg(RegNode)
        imm = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri16(op1, self.reg.eval(ctx), 0, self.imm.eval(ctx, 16, signed=True))

    return MyNode


def form_ririim1(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        imm1 = form_arg(ImmNode)
        reg2 = form_arg(RegNode)
        imm2 = form_arg(ImmNode)
        imm3 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rrrrr(op1, op2, self.reg1.eval(ctx), self.imm1.eval(ctx, 5), self.reg2.eval(ctx), self.imm2.eval(ctx, 5), self.imm3.eval(ctx, 5, offset=1))

    return MyNode


def form_rirzim1(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        imm1 = form_arg(ImmNode)
        reg2 = form_arg(RegNode)
        imm2 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rrrrr(op1, op2, self.reg1.eval(ctx), self.imm1.eval(ctx, 5), self.reg2.eval(ctx), 0, self.imm2.eval(ctx, 5, offset=1))

    return MyNode


def form_rrs11r(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)
        imm = form_arg(ImmNode)
        reg3 = form_arg(RegNode)

        def assemble(self, ctx):
            return asm_rri11r(op1, self.reg1.eval(ctx), self.reg2.eval(ctx), self.imm.eval(ctx, 11, signed=True), self.reg3.eval(ctx))

    return MyNode


def form_rrz11r(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)
        reg3 = form_arg(RegNode)

        def assemble(self, ctx):
            return asm_rri11r(op1, self.reg1.eval(ctx), self.reg2.eval(ctx), 0, self.reg3.eval(ctx))

    return MyNode


def form_rii11im1(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        imm1 = form_arg(ImmNode)
        imm2 = form_arg(ImmNode)
        imm3 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri11r(op1, self.reg1.eval(ctx), self.imm1.eval(ctx, 5), self.imm2.eval(ctx, 11), self.imm3.eval(ctx, 5, offset=1))

    return MyNode


def form_riz11z(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        imm1 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri11r(op1, self.reg1.eval(ctx), self.imm1.eval(ctx, 5), 0, 0)

    return MyNode


def form_rio11z(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        imm1 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri11r(op1, self.reg1.eval(ctx), self.imm1.eval(ctx, 5), 1, 0)

    return MyNode


def form_rzriim1(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)
        imm2 = form_arg(ImmNode)
        imm3 = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rrrrr(op1, op2, self.reg1.eval(ctx), 0, self.reg2.eval(ctx), self.imm2.eval(ctx, 5), self.imm3.eval(ctx, 5, offset=1))

    return MyNode


def form_rzrzf(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)

        def assemble(self, ctx):
            return asm_rrrrr(op1, op2, self.reg1.eval(ctx), 0, self.reg2.eval(ctx), 0, 0x1f)

    return MyNode


def form_rrr(op1, op2, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)
        reg3 = form_arg(RegNode)

        def assemble(self, ctx):
            return asm_rrr(op1, op2, self.reg1.eval(ctx), self.reg2.eval(ctx), self.reg3.eval(ctx))

    return MyNode


def form_rri16(op1, name):
    @form_node
    class MyNode(FormNode, InsnNode):
        symbol = Symbol(name)
        reg1 = form_arg(RegNode)
        reg2 = form_arg(RegNode)
        imm = form_arg(ImmNode)

        def assemble(self, ctx):
            return asm_rri16(op1, self.reg1.eval(ctx), self.reg2.eval(ctx), self.imm.eval(ctx, 16, signed=True))

    return MyNode


class SourceNode(AlternativesNode):
    alternatives = [
        RegisterNode,
        ConstNode,
        LabelNode,
        form_orz12(0x00, 0x0, 0x00, 'rcmd'),
        form_osri12(0x00, 0x0, 0x01, 'error'),
        form_ozz12(0x00, 0x0, 0x02, 'pong'),
        form_osri12(0x00, 0x0, 0x03, 'xycmd'),
        form_osri12(0x00, 0x0, 0x04, 'texcmd'),
        form_osri12(0x00, 0x0, 0x05, 'flcmd'),
        form_osri12(0x00, 0x0, 0x06, 'fzcmd'),
        form_osri12(0x00, 0x0, 0x07, 'ogcmd'),
        form_ozi12(0x00, 0x0, 0x08, 'b'),
        form_ori12(0x00, 0x0, 0x09, 'bl'),
        form_ori12(0x00, 0x0, 0x0a, 'bi'),
        form_orz12(0x00, 0x0, 0x0a, 'br'),
        form_rii12(0x00, 0x2, 'bbs'),
        form_rii12(0x00, 0x3, 'bbc'),
        form_rri12(0x00, 0x4, 'be'),
        form_rri12(0x00, 0x5, 'bne'),
        form_rri12(0x00, 0x6, 'bg'),
        form_rri12(0x00, 0x7, 'ble'),
        form_rri12(0x00, 0x8, 'st'),
        form_rri12(0x00, 0x9, 'ld'),
        form_ri7i12(0x01, 0x0, 'bei'),
        form_ri7i12(0x01, 0x1, 'bnei'),
        form_ri7i12(0x01, 0x2, 'bgi'),
        form_ri7i12(0x01, 0x3, 'blei'),
        form_ririim1(0x02, 0, 'mb'),
        form_rirzim1(0x02, 0, 'dep'),   # alias
        form_ririim1(0x02, 1, 'mbc'),
        form_rzriim1(0x02, 1, 'extr'),  # alias
        form_rzrzf(0x02, 1, 'mov'),     # alias
        form_rii11im1(0x03, 'mbi'),
        form_riz11z(0x03, 'clrb'),      # alias
        form_rio11z(0x03, 'setb'),      # alias
        form_rzi16(0x04, 'li'),
        form_rri16(0x06, 'ai'),
        form_rri16(0x07, 'rsi'),
        form_rrs11r(0x08, 'a'),
        form_rrz11r(0x08, 'ar'),        # alias
        form_rrs11r(0x09, 's'),
        form_rrz11r(0x09, 'sr'),        # alias
        form_rrz11r(0x0a, 'arm50'),
        form_rrz11r(0x0b, 'sign'),
    ]


class Context:
    def __init__(self):
        self.pos = 0
        self.names = {}

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.source) as f:
        source = [SourceNode(node) for node in read_file(f)]

    # First pass.
    ctx = Context()
    for node in source:
        node.first(ctx)

    # Second pass.
    data = b''.join(
        node.assemble(ctx).to_bytes(4, 'little')
        for node in source
        if isinstance(node, InsnNode)
    )

    # Output.
    with open(args.output, 'wb') as f:
        f.write(data)
