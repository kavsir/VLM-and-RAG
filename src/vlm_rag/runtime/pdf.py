"""Optional PDFium renderer for normalized, top-left-origin evidence crops."""

import hashlib
import importlib
import io
import math

from vlm_rag.physical_ir.models import BoundingBox
from vlm_rag.vlm.evidence import ResolvedVisualEvidence
from vlm_rag.vlm.models import RenderSpecification


class PDFiumRenderer:
    def render(
        self,
        pdf_bytes: bytes,
        *,
        page_index: int,
        bbox: object,
        specification: RenderSpecification,
    ) -> ResolvedVisualEvidence:
        if not isinstance(bbox, BoundingBox) or bbox.x0 >= bbox.x1 or bbox.y0 >= bbox.y1:
            raise ValueError("render requires a non-empty normalized bounding box")
        try:
            pdfium = importlib.import_module("pypdfium2")
            importlib.import_module("PIL.Image")
        except ImportError:
            raise ValueError("PDF rendering requires: uv sync --locked --extra pdf") from None
        with pdfium.PdfDocument(pdf_bytes) as document:
            if not 0 <= page_index < len(document):
                raise ValueError("render page out of range")
            page = document[page_index]
            try:
                width, height = page.get_size()
                scale = specification.dpi / 72.0
                if math.ceil(width * scale) * math.ceil(height * scale) > 25_000_000:
                    raise ValueError("page render exceeds 25 million pixel budget")
                bitmap = page.render(scale=scale, rev_byteorder=True)
                try:
                    image = bitmap.to_pil()
                    try:
                        # Crop the displayed bitmap: handles PDF page rotation consistently.
                        bounds = (
                            math.floor(image.width * bbox.x0 / 1000),
                            math.floor(image.height * bbox.y0 / 1000),
                            math.ceil(image.width * bbox.x1 / 1000),
                            math.ceil(image.height * bbox.y1 / 1000),
                        )
                        with image.crop(bounds).convert("RGB") as crop:
                            output = io.BytesIO()
                            crop.save(output, format="PNG")
                            data = output.getvalue()
                            return ResolvedVisualEvidence(
                                data=data,
                                sha256=hashlib.sha256(data).hexdigest(),
                                byte_size=len(data),
                                media_type="image/png",
                                width=crop.width,
                                height=crop.height,
                            )
                    finally:
                        image.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
