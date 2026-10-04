"""TensorCore 发射模型：按本节的规则逐 bundle 重放一段动态执行的清单，给出每次 LCC 读数的标量发射时刻。

输入是实际执行的 bundle 序列（循环要按迭代展开），每个 bundle 是 tpuasm 清单中 `{ ... }` 的文本。模型不处理 DMA 与 vwait：它们的时间取决于数据通路，由第二章的开销模型给出。标量一侧只建模了 sld 的两条规则，其余标量指令都按下一周期可用处理。
"""
import re

VIF_ENTRIES = 20  # 标量侧有 20 项未释放时，任何 bundle 都不能标量发射。
RELEASE = 10  # 一项在向量发射后 10 个周期才在标量侧释放。
MEMORY = ('vld', 'vst', 'cld')  # 访问内存的 bundle 最早在标量发射后 2 个周期向量发射，其余 1 个周期。
RESULT_LATENCY = {'vmul.8x128.f32': 2, 'vrot.slane.down.8x128.u32': 2}  # 写 TC VREG 的指令：结果在几个周期后可用，未列出的为 1。
# 提交—取回通路：结果进入哪个队列、发射后多少周期可以取回。
PUSH_LATENCY = {'erf': 7, 'mrf': 83, 'trf': 6, 'crf': 53, 'v2sf': 42}
# XLU 的其他操作同样进入 trf 队列，但每次提交产生一个结果：跨 lane 归约 79 个周期、lane 循环移位与重排 69 个周期后可以取回。
# vmax.index.xlane 与 vmin.xlane 由第一章第 14 节测得。
XLU_LATENCY = {'vadd.xlane': 79, 'vmax.xlane': 79, 'vmax.index.xlane': 79, 'vmin.xlane': 79, 'vrot.': 69, 'vperm.': 69}
# 同一个队列上，相邻两条提交（push）或取回（pop）指令的最小发射间隔，未列出的为 1。
INTERVAL = {'push erf': 2, 'push mrf': 8, 'pop mrf': 8, 'push trf': 8, 'pop trf': 8, 'push crf': 2, 'pop crf': 2}
# 同一单元上相邻两条指令的最小间隔，按助记符前缀匹配：vrng 每条让生成器走 8 步（第四章第 6 节）。
UNIT_INTERVAL = {'vrng': 8}
SLD_INTERVAL = 4  # 相邻两条 sld 至少相隔 4 个周期（第 3 节）。
SLD_LATENCY = 4  # sld 的结果 4 个周期后才能被标量指令使用（第 3 节）。

def parse(text: str) -> list[list[str]]:
    """把清单文本拆成 bundle，每个 bundle 是若干 `槽位: 指令` 字符串。"""
    return [[item.strip() for item in body.split(';') if item.strip()] for body in re.findall(r'\{(.*?)\}', text)]

def decode(item: str) -> tuple[str, list[str]]:
    """`槽位: 助记符 操作数, ...` → (助记符, 操作数列表)。"""
    mnemonic, _, operands = item.split(':', 1)[1].strip().partition(' ')
    return mnemonic, [operand.strip() for operand in operands.split(',') if operand.strip()]

def kind(queue: str) -> str:
    return re.sub(r'\d+$', '', queue)

def replay(program: list[list[str]]) -> dict[int, int]:
    """返回 srdreg.lcclo 各目的寄存器（最后一次）读数的标量发射时刻。"""
    scalar = 0  # 当前 bundle 的标量发射时刻
    vector_free = 0  # 向量侧可以发射下一个 bundle 的最早时刻
    releases: list[int] = []  # 尚未在标量侧释放的 VIF 项的释放时刻
    ready: dict[str, int] = {}  # TC VREG 的结果可用时刻
    queues: dict[str, list[int]] = {}  # 各队列中结果的可取回时刻
    last: dict[str, int] = {}  # 各队列上一次提交或取回的时刻
    scalar_ready: dict[str, int] = {}  # sld 写入的标量寄存器的可用时刻
    last_sld = -99
    reads = {}
    for items in program:
        pending = sorted(release for release in releases if release > scalar)
        if len(pending) >= VIF_ENTRIES:
            scalar = pending[len(pending) - VIF_ENTRIES]
        releases = [release for release in releases if release > scalar]
        scalar_items = [decode(item) for item in items if item.startswith(('s0:', 's1:'))]
        for mnemonic, operands in scalar_items:
            if mnemonic == 'sld':
                scalar = max(scalar, last_sld + SLD_INTERVAL)
            # 读取尚未就绪的标量寄存器时，整个 bundle 等待。
            sources = re.findall(r'\bs\d+\b', ' '.join(operands[1:])) + re.findall(r'\[smem:(s\d+)\]', ' '.join(operands[:1]))
            scalar = max([scalar, *(scalar_ready.get(source, 0) for source in sources)])
        for mnemonic, operands in scalar_items:
            if mnemonic == 'spop':
                scalar = max(scalar, queues[operands[1]].pop(0))
        for mnemonic, operands in scalar_items:
            if mnemonic == 'sld':
                last_sld = scalar
                scalar_ready[operands[0]] = scalar + SLD_LATENCY
            elif operands and re.fullmatch(r's\d+', operands[0]):
                scalar_ready.pop(operands[0], None)
            if mnemonic == 'srdreg.lcclo':
                reads[int(operands[0][1:])] = scalar
        vector = [item for item in items if not item.startswith(('s0:', 's1:'))]
        fence = 's0: sfence' in items
        if not vector and not fence:
            scalar += 1
            continue
        events = []  # (提交、取回或单元, 队列或单元名)
        issue = max(vector_free, scalar + 1 + any(decode(item)[0].startswith(MEMORY) for item in vector))
        for item in vector:
            mnemonic, operands = decode(item)
            if mnemonic.startswith('vpop'):
                events.append(('pop', operands[1]))
                issue = max(issue, queues[operands[1]][0])
            elif operands and kind(operands[0]) in PUSH_LATENCY:
                events.append(('push', operands[0]))
            unit = next((prefix for prefix in UNIT_INTERVAL if mnemonic.startswith(prefix)), None)
            if unit:
                events.append(('unit', unit))
            issue = max([issue, *(ready.get(source, 0) for source in operands[1:])])
        for event, queue in events:
            issue = max(issue, last.get(f'{event} {queue}', -99) + (UNIT_INTERVAL[queue] if event == 'unit' else INTERVAL.get(f'{event} {kind(queue)}', 1)))
        for item in vector:
            mnemonic, operands = decode(item)
            if mnemonic.startswith('vpop'):
                queues[operands[1]].pop(0)
                ready[operands[0]] = issue + 1
            elif operands and kind(operands[0]) in PUSH_LATENCY:
                # 转置要收齐 16 个 TC VREG，最后一次提交（.end）之后才产生 16 个结果。
                count = 0 if mnemonic.startswith('vxpose') and '.end' not in mnemonic else 16 if mnemonic.startswith('vxpose') else 1
                latency = next((value for prefix, value in XLU_LATENCY.items() if mnemonic.startswith(prefix)), PUSH_LATENCY[kind(operands[0])])
                queues.setdefault(operands[0], []).extend([issue + latency] * count)
            elif operands and re.fullmatch(r'v\d+', operands[0]):
                ready[operands[0]] = issue + RESULT_LATENCY.get(mnemonic, 1)
        for event, queue in events:
            last[f'{event} {queue}'] = issue
        vector_free = issue + 1
        releases.append(issue + RELEASE)
        # sfence 所在 bundle 在向量侧按序占一个位置；下一 bundle 等所有项（包括它自己）在标量侧释放。
        scalar = max(releases) if fence else scalar + 1
    return reads
