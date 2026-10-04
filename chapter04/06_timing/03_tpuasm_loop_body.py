"""解释第 2 个实验中均匀分布、指数分布与 Gumbel 分布每个 TC VREG 的周期数：取出编译器生成的循环体，用 LCC 实测它本身的周期数，再用第三章第 5 节的发射模型分别在不考虑与考虑 vrng 发射间隔时预测。"""
import tpu_init
tpu_init.initialise_one_chip()

from pathlib import Path
import re
import sys

from generation_common import METHODS, build
import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'chapter03' / '05_issue_model'))
import issue_model

# (方法, 每次生成的行数)：前三个与第 2 个实验相同，最后一个只改每次的行数。
CASES = (('vrng → 均匀分布（stateful_uniform）', 64), ('vrng → 指数分布', 64), ('vrng → Gumbel 分布', 64), ('vrng → 指数分布', 256))
ITERATIONS = 4  # 循环体重复几次
# 循环体会改写 LccProbe 保存标量寄存器用的 v20–v30：setup 先把它们存进 TC VMEM，最后一次读数之后读回。
SETUP = tpuasm_tools.spill_saved() + bundle('va0: setrngseed v10') + bundle('misc: vnop') * 16
RESTORE = tpuasm_tools.reload_saved()
END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)

def loop_body(compiled) -> list[str]:
    """kernel 段中 pl.loop 的循环体（从循环标号到回跳分支之后的延迟槽），去掉标量指令与编码约束。"""
    lines = [line.split('#')[0].rstrip() for line in tpuasm_tools.kernel_listing(compiled, pallas_only=True).splitlines()]
    text = re.sub(r'\s*;\s*\.encoding \{[^}]*\}', '', ' '.join(line for line in lines if line.strip()))
    label, = re.findall(r'sbr\.rel (L_\w+)', text[text.index('setrngseed'):])
    loop = text[text.index(f'{label}:') + len(label) + 1:]
    bundles = [body.strip() for body in re.findall(r'\{(.*?)\}', loop)]
    branch = next(index for index, body in enumerate(bundles) if 'sbr.rel' in body)
    # 标量指令（计数、回跳）留在载体之外：它们只占标量槽，去掉后原来的 bundle 仍占一个周期。
    return [' ; '.join(item.strip() for item in body.split(';') if not item.strip().startswith(('s0:', 's1:'))) for body in bundles[:branch + 2]]

def predict(body: list[str], tiles: int, with_vrng: bool) -> float:
    saved = dict(issue_model.UNIT_INTERVAL)
    if not with_vrng:
        issue_model.UNIT_INTERVAL.clear()
    reads = issue_model.replay(issue_model.parse(read_lcc(20) + ''.join(bundle(text) for text in body) * ITERATIONS + END))
    issue_model.UNIT_INTERVAL.update(saved)
    return (reads[21] - reads[20]) / (tiles * ITERATIONS)

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for name, rows in CASES:
        _, generate = METHODS[name]
        tiles = rows // 8
        mesh, draw = build(rows, generate, 8)
        body = loop_body(tpuasm_tools.compile(draw, mesh=mesh))
        vrng = [index for index, text in enumerate(body) if 'vrng' in text]
        print(f'## {name}，每次 {rows} 行')
        print(f'  循环体 {len(body)} 个 bundle，生成 {tiles} 个 TC VREG；vrng 在第 {vrng} 个 bundle，相邻两条相隔 {[b - a for a, b in zip(vrng, vrng[1:])]}')
        measured = {tuple(row) for row in probe.run(read_lcc(20) + ''.join(bundle(text) for text in body) * ITERATIONS + END + RESTORE, setup=SETUP).tolist()}
        (r1, _), = measured
        print(f'  LCC 实测：循环体重复 {ITERATIONS} 次，R1 − R0 = {r1}，每次 {r1 / ITERATIONS:.1f} 个周期，每个 TC VREG {r1 / (tiles * ITERATIONS):.2f} 个周期')
        print(f'  发射模型：不考虑 vrng 的间隔 {predict(body, tiles, False):.2f}，考虑 8 个周期的间隔 {predict(body, tiles, True):.2f}')
        if rows != 64:
            continue
        print('  循环体：')
        for index, text in enumerate(body):
            print(f'    [{index:2d}] {{ {text} }}')

if __name__ == '__main__':
    main()
