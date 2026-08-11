"""Constants for the classic WEKA multiple-instance datasets."""

from pathlib import Path

DEFAULT_CLASSIC_MIL_DATA_DIR = Path("data/multi_instance/classic_mil")
CLASSIC_MIL_ARCHIVE_NAME = "multi-instance.zip"
CLASSIC_MIL_RELEASE = "classic-mil-v1"
CLASSIC_MIL_ARCHIVE_URLS = (
    "https://github.com/jmachadofreitas/tiny-datasets/releases/download/"
    f"{CLASSIC_MIL_RELEASE}/multi-instance.zip",
    "https://sourceforge.net/projects/weka/files/datasets/multi-instance/"
    "multi-instance.zip/download",
)
CLASSIC_MIL_ARCHIVE_SIZE_BYTES = 7_520_161
CLASSIC_MIL_ARCHIVE_SHA256 = "8190597a5edcc1167a2fe08d26f393edf232287c3e3271a03dc7bdee1ea9056d"
CLASSIC_MIL_ARCHIVE_ROOT = "multi_instance"

CLASSIC_MIL_DATASETS = (
    "component",
    "eastwest",
    "elephant",
    "fox",
    "function",
    "musk1",
    "musk2",
    "mutagenesis3_atoms",
    "mutagenesis3_bonds",
    "mutagenesis3_chains",
    "process",
    "suramin",
    "tiger",
    "trx",
    "westeast",
)
