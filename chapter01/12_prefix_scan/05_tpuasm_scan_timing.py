"""几种前缀扫描的计算部分各要多少周期：把编译器生成的计算 bundle 原样插进载体，用 LCC 实测，并与第三章第 6 节的发射模型比较。"""
import tpu_init
tpu_init.initialise_one_chip()

from pathlib import Path
import sys

import numpy as np

import scan_common
import tpuasm_tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'chapter03' / '06_issue_model'))
import issue_model

def main() -> None:
    x = np.random.default_rng(0).integers(-100, 100, (8, 128)).astype(np.float32)
    listings = {
        '沿 lane，Hillis–Steele': scan_common.run(scan_common.hillis_steele_lanes, x)[1],
        '沿 sublane，前缀和': scan_common.run(scan_common.hillis_steele_sublanes, x)[1],
        '沿 sublane，后缀和': scan_common.run(scan_common.suffix_sublanes, x)[1],
        '沿 sublane，逐行广播（stride=0）': scan_common.run_row_serial('stride=0', x)[1],
    }
    probe = tpuasm_tools.LccProbe()
    for name, listing in listings.items():
        section = tpuasm_tools.compute_section(listing)
        _, measured = probe.time_section(section)
        reads = issue_model.replay(issue_model.parse(tpuasm_tools.section_program(section)))
        print(f'{name}：{len(section)} 个 bundle；实测 R2 − R0 = {measured}，模型 {reads[22] - reads[20]}')

if __name__ == '__main__':
    main()
