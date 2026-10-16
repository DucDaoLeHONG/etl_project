from .dll_imports import DllImportsPipeline
from .hash_samples import HashSamplesPipeline
from .pe_header import PeHeaderPipeline
from .pe_headers import PeHeadersPipeline
from .pe_multiview import PeMultiviewPipeline
from .pe_section import PeSectionPipeline
from .xpe_guard import XpeGuardPipeline

# Order matters: pe_multiview consumes the outputs of dll_imports, pe_header and pe_section.
ALL = {
    "hash_samples": HashSamplesPipeline,
    "xpe_guard": XpeGuardPipeline,
    "pe_headers": PeHeadersPipeline,
    "dll_imports": DllImportsPipeline,
    "pe_header": PeHeaderPipeline,
    "pe_section": PeSectionPipeline,
    "pe_multiview": PeMultiviewPipeline,
}
