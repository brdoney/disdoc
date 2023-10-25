from os.path import isdir
from typing import List, Optional, Tuple, cast
import fitz
import langchain.schema
import uuid
import os


def get_cropped_image(search_results: List[fitz.Rect], page: fitz.Page) -> str:
    overall: fitz.Rect = search_results[0]
    for rect in search_results[1:]:
        overall |= rect

    # Set cropbox so no content shows in padded area
    page.set_cropbox(overall)

    # Add padding so that we'll have some space after clipping
    padding = 10
    padding_box = fitz.Rect(overall)
    padding_box.x0 -= padding
    padding_box.y0 -= padding
    padding_box.x1 += padding
    padding_box.y1 += padding
    pixels = page.get_pixmap(clip=padding_box)  # type: ignore

    image_name = f"./image-{uuid.uuid4()}.png"
    pixels.save(image_name)

    return image_name


__cached_textpages = {}
__cached_images = {}


def __search_pdf(
    search: str, pdf: fitz.Document, document: langchain.schema.Document
) -> Optional[Tuple[fitz.Page, List[fitz.Quad]]]:
    page_num = document.metadata["page"]
    page = pdf.load_page(page_num)

    key = document.metadata["source"] + str(page_num)
    if key in __cached_textpages:
        tp = __cached_textpages[key]
    else:
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


def pdf_image(doc_name: str, document: langchain.schema.Document) -> Optional[str]:
    key = document.metadata["source"] + document.page_content
    if key in __cached_images:
        return __cached_images[key]

    with fitz.open(  # type: ignore
        f"/home/grads/brendandoney/Thesis/privateGPT/source_documents/{doc_name}"
    ) as pdf:
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

        if not os.path.isdir("./pdf_images"):
            os.mkdir("./pdf_images")

        image_name = f"./pdf_images/image-{uuid.uuid4()}.png"
        page.get_pixmap().save(image_name)  # type: ignore

        __cached_images[key] = image_name

        return image_name
