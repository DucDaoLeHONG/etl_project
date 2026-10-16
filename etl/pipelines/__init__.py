from .figshare_dll import FigshareDllPipeline
from .figshare_header import FigshareHeaderPipeline
from .figshare_multiview import FigshareMultiviewPipeline
from .figshare_section import FigshareSectionPipeline
from .kaggle_header import KaggleHeaderPipeline
from .pe_malware_dataset1 import PeMalwareDataset1Pipeline
from .xpe_guard import XpeGuardPipeline
from .zenodo_hashes import ZenodoHashesPipeline

# Order matters: figshare_multiview consumes the staging tables of the three figshare view pipelines.
ALL = {
    "zenodo_hashes": ZenodoHashesPipeline,
    "xpe_guard": XpeGuardPipeline,
    "kaggle_header": KaggleHeaderPipeline,
    "figshare_dll": FigshareDllPipeline,
    "figshare_header": FigshareHeaderPipeline,
    "figshare_section": FigshareSectionPipeline,
    "figshare_multiview": FigshareMultiviewPipeline,
    "pe_malware_dataset1": PeMalwareDataset1Pipeline,
}
