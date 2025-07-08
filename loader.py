# loader.py
from model import BookTran
from utils import get_clean_text_from_html
from vector import get_vector
from tortoise.expressions import Q
import pandas as pd
import numpy as np

async def load_book_df(limit=200):
    books = await BookTran.filter(status="publish").limit(limit)
    data = []
    for b in books:
        tag_text = b.tag.replace(",", " ") if b.tag else ""
        name_text = b.name or ""
        des_text = get_clean_text_from_html(b.des) or ""
        combined_text = f"{name_text} " * 3 + f"{tag_text} " * 2 + des_text
        vec = get_vector(combined_text)
        data.append({
            "bookID": b.bookID,
            "name": name_text,
            "tag": tag_text,
            "des": des_text,
            "vector": vec
        })
    return pd.DataFrame(data)

async def load_book_df_by_one(book_id, limit=200):
    books = await BookTran.filter(
        bookID=book_id, status="publish"
    ).only("bookID", "name", "tag", "des").limit(limit)

    data = []
    for b in books:
        tag_text = b.tag.replace(",", " ") if b.tag else ""
        name_text = b.name or ""
        des_text = get_clean_text_from_html(b.des) or ""
        combined_text = f"{name_text} " * 3 + f"{tag_text} " * 2 + des_text
        vec = get_vector(combined_text)
        data.append({
            "bookID": b.bookID,
            "name": name_text,
            "tag": tag_text,
            "des": des_text,
            "vector": vec
        })
    return pd.DataFrame(data)

async def load_book_df_with_not_in(exclude_ids: list[str], limit=200):
    books = await BookTran.filter(~Q(bookID__in=exclude_ids), status="publish").only(
        "bookID", "name", "tag", "des").limit(limit)
    data = []
    for b in books:
        tag_text = b.tag.replace(",", " ") if b.tag else ""
        name_text = b.name or ""
        des_text = get_clean_text_from_html(b.des or "")
        combined_text = f"{name_text} " * 3 + f"{tag_text} " * 2 + des_text
        vec = get_vector(combined_text)
        data.append({
            "bookID": b.bookID,
            "name": name_text,
            "tag": tag_text,
            "des": des_text,
            "vector": vec
        })
    return pd.DataFrame(data)

async def load_book_df_title(exclude_ids: list[str], limit=200):
    books = await BookTran.filter(~Q(bookID__in=exclude_ids), status="publish").only(
        "bookID", "name", "tag", "des", "title").limit(limit)
    data = []
    for b in books:
        tag_text = b.tag.replace(",", " ") if b.tag else ""
        name_text = b.name or ""
        title_text = b.title or ""
        combined_text = f"{name_text} {tag_text} {title_text}"
        vec = get_vector(combined_text)
        data.append({
            "bookID": b.bookID,
            "name": name_text,
            "tag": tag_text,
            "title": title_text,
            "vector": vec
        })
    return pd.DataFrame(data)

async def load_all_books_and_vectors(exclude_ids: list[str]):
    books = await BookTran.filter(~Q(bookID__in=exclude_ids), status="publish").only(
        "bookID", "name", "tag", "des", "title")
    vectors = []
    meta = []
    for b in books:
        tag_text = b.tag.replace(",", " ") if b.tag else ""
        name_text = b.name or ""
        title_text = b.title or ""
        combined_text = f"{name_text} {tag_text} {title_text}"
        vec = get_vector(combined_text)
        vectors.append(vec)
        meta.append({
            "bookID": b.bookID,
            "name": name_text,
            "tag": tag_text,
            "title": title_text
        })
    return np.array(vectors, dtype=np.float32), meta