from pydantic import BaseModel


class ChunkContext(BaseModel):
    previous_chunks: list[str] | None = None
    next_chunks: list[str] | None = None
    document_title: str | None = None
    section_headers: list[str] | None = None
    hierarchy_path: str | None = None
