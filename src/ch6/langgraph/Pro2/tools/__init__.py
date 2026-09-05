from .search import search_web, search_arxiv
from .knowledge import search_knowledge_base
from .file_ops import save_report, list_reports

ALL_TOOLS = [
    search_web,
    search_arxiv,
    search_knowledge_base,
    save_report,
    list_reports,
]