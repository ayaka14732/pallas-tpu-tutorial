"""用 LCC 找出模型缺少的规则：把一段清单的前 p 个 bundle 逐个加长，比较实测与模型，差值在哪个 bundle 上跳变，缺的规则就在那里。由此得到 TC VMEM 写后读的等待，再单独测出它的参数。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc
import issue_model

END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
# 一段把中间结果存进 TC VMEM、稍后再读回的清单。0x80 起的行在载体的 TC VMEM 中没有别的用途。
SECTION = (
    'va0: vadd.8x128.s32 v11, 1, v10',
    'vst: vst.8x128 [vmem:0x80], v11',
    'va0: vadd.8x128.s32 v12, 2, v10',
    'va0: vadd.8x128.s32 v13, v12, v11',
    'vld: vld.8x128 v14, [vmem:0x80]',
    'va0: vadd.8x128.s32 v15, v14, v13',
    'vst: vst.8x128 [vmem:0x88], v15',
    'vld: vld.8x128 v16, [vmem:0x90]',
    'va0: vadd.8x128.s32 v17, v16, v15',
    'vld: vld.8x128 v18, [vmem:0x88]',
    'va0: vadd.8x128.s32 v19, v18, v17',
    'vst: vst.8x128 [vmem:0x80], v19',
)

def measure(probe: tpuasm_tools.LccProbe, body: str) -> int:
    """R0、body、R1、sfence、R2 的 R2 − R0；8 次运行必须相同。"""
    (_, second), = {tuple(row) for row in probe.run(read_lcc(20) + body + END).tolist()}
    return second

def predict(body: str, store_load: int) -> int:
    """模型给出的 R2 − R0；store_load 是写后读的等待，0 表示没有这条规则。"""
    saved, issue_model.STORE_LOAD = issue_model.STORE_LOAD, store_load
    reads = issue_model.replay(issue_model.parse(read_lcc(20) + body + END))
    issue_model.STORE_LOAD = saved
    return reads[22] - reads[20]

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    print('## 逐个 bundle 加长：前 p 个 bundle 的 R2 − R0')
    previous = 0
    for count in range(1, len(SECTION) + 1):
        body = ''.join(bundle(text) for text in SECTION[:count])
        measured, without, with_rule = measure(probe, body), predict(body, 0), predict(body, issue_model.STORE_LOAD)
        print(f'  p = {count:2d}：实测 {measured}，没有写后读规则的模型 {without}，差 {measured - without}（{measured - without - previous:+d}），加上规则的模型 {with_rule}    {SECTION[count - 1]}')
        previous = measured - without

    store = bundle('vst: vst.8x128 [vmem:0x80], v10')
    loads = {
        '读同一个 tile（0x80）': bundle('vld: vld.8x128 v11, [vmem:0x80]'),
        '读错开 4 行的 tile（0x84）': bundle('vld: vld.8x128 v11, [vmem:0x84]'),
        '读相邻的 tile（0x88）': bundle('vld: vld.8x128 v11, [vmem:0x88]'),
    }
    print('## vst 之后隔 d − 1 个 vnop 再 vld，R2 − R0')
    for name, load in loads.items():
        print(f'  {name}：' + '，'.join(f'd = {distance}：{measure(probe, store + bundle("misc: vnop") * (distance - 1) + load)}' for distance in (1, 2, 4, 6, 7, 8, 10)))
    print('## 其他排列，R2 − R0（实测，模型）')
    same = loads['读同一个 tile（0x80）']
    cases = {
        'vst 与读同一个 tile 的 vld 在同一个 bundle': bundle('vst: vst.8x128 [vmem:0x80], v10 ; vld: vld.8x128 v11, [vmem:0x80]'),
        '8 组（vst、读同一个 tile 的 vld）': (store + same) * 8,
        '8 组（vst、读相邻 tile 的 vld）': (store + loads['读相邻的 tile（0x88）']) * 8,
        '8 条 vst，再 8 条读同一个 tile 的 vld': store * 8 + same * 8,
        '8 组（读同一个 tile 的 vld、vst）': (same + store) * 8,
    }
    for name, body in cases.items():
        print(f'  {name}：{measure(probe, body)}，{predict(body, issue_model.STORE_LOAD)}')

if __name__ == '__main__':
    main()
