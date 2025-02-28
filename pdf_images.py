from typing import cast
import fitz  # pyright: ignore[reportMissingTypeStubs]
import langchain.schema
import uuid
import os
import pickle
from pathlib import Path

__CACHE_FOLDER = "./cache"

# Cached in-memory only
__cached_textpages = {}
# Cached on the disk in the `CACHE_FOLDER`
__cached_images = {}
__CACHED_IMAGES_PATH = f"{__CACHE_FOLDER}/image-cache.pkl"


def save_image_cache() -> None:
    with open(__CACHED_IMAGES_PATH, "wb") as f:
        pickle.dump(__cached_images, f)


def load_image_cache() -> None:
    global __cached_images
    try:
        with open(__CACHED_IMAGES_PATH, "rb") as f:
            __cached_images = pickle.load(f)
    except FileNotFoundError:
        pass


def get_cropped_image(search_results: list[fitz.Rect], page: fitz.Page) -> str:
    overall: fitz.Rect = search_results[0]
    for rect in search_results[1:]:
        overall |= rect

    # Set cropbox so no content shows in padded area
    page.set_cropbox(overall)  # pyright: ignore[reportUnknownMemberType]

    # Add padding so that we'll have some space after clipping
    padding = 10
    padding_box = fitz.Rect(overall)
    padding_box.x0 -= padding
    padding_box.y0 -= padding
    padding_box.x1 += padding
    padding_box.y1 += padding
    pixels: fitz.Pixmap = page.get_pixmap(clip=padding_box)

    if not os.path.isdir(__CACHE_FOLDER):
        os.mkdir(__CACHE_FOLDER)

    image_name = f"{__CACHE_FOLDER}/image-{uuid.uuid4()}.png"
    pixels.save(image_name)

    return image_name


def __search_pdf(
    search: str, pdf: fitz.Document, document: langchain.schema.Document
) -> tuple[fitz.Page, list[fitz.Quad]] | None:
    page_num = document.metadata["page"]
    page = pdf.load_page(page_num)

    key = document.metadata["source"] + str(page_num)
    if key in __cached_textpages:
        tp = __cached_textpages[key]
    else:
        # Preserve whitespace is overkill b/c converting to text is lossy
        # Ligatures not necessary b/c users will never put ligatures in their questions
        tp = page.get_textpage(
            flags=fitz.TEXT_DEHYPHENATE
            # | fitz.TEXT_PRESERVE_WHITESPACE
            # | fitz.TEXT_PRESERVE_LIGATURES,
        )
        __cached_textpages[key] = tp

    rects = tp.search(search)
    if len(rects) > 0:
        return page, rects
    return None


def pdf_image(
    doc_path: Path, document: langchain.schema.Document
) -> tuple[Path, bool] | None:
    """Get a PDF image from a document - either from the cache or by generating it -
    and return a URL to the image along with whether it was in the cache. If the
    excerpt could not be found in the document, returns None.

    Args:
        doc_path: path to the file the document is from
        document: the document (the relevant excerpt) we are interested in

    Returns:
        Either a path to the document
    """
    key = document.metadata["source"] + document.page_content
    if key in __cached_images:
        return Path(__cached_images[key]), True

    with fitz.open(doc_path) as pdf:
        pdf = cast(fitz.Document, pdf)

        # print(page.get_textpage().extractText())
        # rects = page.get_textpage().search(search)
        res = __search_pdf(document.page_content, pdf, document)
        if res is None:
            print("Couldn't find it")
            return None
        page, rects = res

        for rect in rects:
            page.add_highlight_annot(rect)

        if not os.path.isdir(__CACHE_FOLDER):
            os.mkdir(__CACHE_FOLDER)

        image_path = Path(f"{__CACHE_FOLDER}/image-{uuid.uuid4()}.png")
        page.get_pixmap(dpi=144).save(image_path)

        __cached_images[key] = str(image_path)

        return image_path, False
