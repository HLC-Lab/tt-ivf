"""Shared colors, RISC names, and legend styling for IVF profiler figures."""

QUERY_READ_COLOR = "#AD9BF4"
READER_COLOR = "#7657E8"
COMPUTE_COLOR = "#F2C500"
WAIT_COLOR = "#7F8C8D"
RESULT_COLOR = "#F45151"

RISC_STAGE = {
    "BRISC": "writer",
    "NCRISC": "reader",
    "TRISC_0": "unpacker",
    "TRISC_1": "math",
    "TRISC_2": "packer",
}


def add_kernel_legend(axis, handles):
    """Use the same compact legend in batch and page timelines."""
    legend = axis.legend(
        handles=handles,
        title="Kernel role",
        loc="upper right",
        ncol=2,
        fontsize=7.5,
        frameon=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#666666")
    return legend
