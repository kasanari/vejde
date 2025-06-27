from collections.abc import Callable
from torch import Tensor, norm


class Lazy:
    def __init__(self, func: Callable[[], str]):
        self.func = func

    def __str__(self) -> str:
        return self.func()


def to_graphviz_bp(
    m_f_to_v: Tensor,
    m_v_to_f: Tensor,
    factors: Tensor,
    variables: Tensor,
    f_to_v: Tensor,
    v_to_f: Tensor,
):
    import graphviz

    output = "digraph G {\n"

    def sum_vector(x: Tensor) -> Tensor:
        return x.sum(dim=-1)

    v_norm = sum_vector(variables)
    f_norm = sum_vector(factors)

    for i, v in enumerate(v_norm):
        output += f'"{i}_v" [label="{v.item():0.2f}", shape=ellipse, color=blue];\n'

    for i, f in enumerate(f_norm):
        output += f'"{i}_f" [label="{f.item():0.2f}", shape=rectangle, color=red];\n'

    for f, v, m in zip(f_to_v, v_to_f, sum_vector(m_f_to_v)):
        output += f'"{int(f)}_f" -> "{int(v)}_v" [label="{m.item():0.2f}"];\n'

    for v, f, m in zip(v_to_f, f_to_v, sum_vector(m_v_to_f)):
        output += f'"{int(v)}_v" -> "{int(f)}_f" [label="{m.item():0.2f}"];\n'

    output += "}"

    g = graphviz.Source(output)
    g.render("frames/frame", format="jpg", engine="sfdp")
    return output.replace("\n", "")


def to_graphviz(
    messages: Tensor,
    senders: Tensor,
    receivers: Tensor,
    variables: Tensor,
    factors: Tensor,
    reverse: bool = False,
):
    output = "digraph G {"

    def vnorm(x: Tensor):
        return norm(x, ord=2, axis=-1)
    #ro = lambda x: round(x, decimals=2)

    v_norm = vnorm(variables)
    f_norm = vnorm(factors)
    m_norm = vnorm(messages)

    def edge_string(v: str, f: str) -> str:
        return f'"{v}_v" -> "{f}_f"' if not reverse else f'"{f}_f" -> "{v}_v"'

    for i, v in enumerate(v_norm):
        output += f'"{i}_v" [label="{v:0.2f}", shape=ellipse, color=blue];'

    for i, f in enumerate(f_norm):
        output += f'"{i}_f" [label="{f:0.2f}", shape=rectangle, color=red];'

    for s, r, m in zip(senders, receivers, m_norm):
        output += f'{edge_string(s, r)} [label="{m:0.2f}"];'

    output += "}"
    return output
