"""把上一个实验的载体原样重新汇编：精确清单与 canonical 清单各一次，各在子进程中运行（60 秒超时），并统计 canonical 编码改变了多少个 bundle 的机器字节。"""
import subprocess
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else None

def child(mode: str) -> None:
    import tpu_init
    tpu_init.initialize_local_chips()
    import importlib

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    import numpy as np
    from tpuasm import assemble_listing, executable_programs, format_assembly, replace_executable_programs

    import tpuasm_tools

    carrier = importlib.import_module('01_tpuasm_cmem_to_cmem')
    exchange, mesh = carrier.build()
    x = jax.device_put(jnp.arange(2 * 8 * 128, dtype=jnp.float32).reshape(2, 8, 128), jax.NamedSharding(mesh, P('device')))
    compiled = tpuasm_tools.compile(exchange, x, mesh=mesh)
    serialized = tpuasm_tools.serialize(compiled)
    (record, index, image), = executable_programs(serialized)
    reassembled = assemble_listing(format_assembly(image, target=tpuasm_tools.TARGET, encoding=mode))
    if mode == 'canonical':
        # 每 512 B 一块，块内 10 个 51 B 的 bundle。
        offsets = [pc * 51 + pc // 10 * 2 for pc in range(len(image) // 512 * 10)]
        changed = sum(image[offset:offset + 51] != reassembled[offset:offset + 51] for offset in offsets)
        print(f'canonical 编码改变了 {changed} / {len(offsets)} 个 bundle 的机器字节', flush=True)
    function = tpuasm_tools.load(replace_executable_programs(serialized, {(record, index): reassembled}), compiled)
    print(f'结果正确：{bool(np.array_equal(np.asarray(function(x)), np.asarray(x)[::-1]))}', flush=True)

def main() -> None:
    for mode in ('exact', 'canonical'):
        try:
            result = subprocess.run([sys.executable, __file__, mode], capture_output=True, text=True, timeout=60)
            print(f'## encoding={mode!r}：' + '；'.join(result.stdout.strip().splitlines()))
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or b'').decode() if isinstance(error.stdout, bytes) else (error.stdout or '')
            print(f'## encoding={mode!r}：' + '；'.join(output.strip().splitlines()) + '；60 秒内没有返回，程序挂起')

if __name__ == '__main__':
    child(MODE) if MODE else main()
