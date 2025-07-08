from fastapi import APIRouter, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from vector import get_vector
from utils import cosine_similarity, get_clean_text_from_html
from model import BookTran, UserHitRead,RecommendSimilar
from datetime import datetime
from tortoise.exceptions import IntegrityError

from loader import (
    load_book_df,
    load_book_df_by_one,
    load_book_df_with_not_in,
    load_book_df_title,
    load_all_books_and_vectors,
)
import numpy as np
import pandas as pd
import faiss

recommend_router = APIRouter()

class RecommendRequest(BaseModel):
    q: str
    topn: int = 10

@recommend_router.post("/recommend/by-keyword")
async def recommend_by_keyword(request: RecommendRequest = Body(...)):
    df = await load_book_df()
    query_vec = get_vector(request.q)
    df["score"] = df["vector"].apply(lambda v: cosine_similarity([query_vec], [v])[0][0])
    result = df.sort_values("score", ascending=False).head(request.topn)
    return result[["bookID", "name", "tag", "des", "score"]].to_dict(orient="records")


@recommend_router.get("/recommend/all-books")
async def recommend_all_books(topn: int = 10):
    # 🔹 Load all books once
    all_books = await BookTran.filter(status="publish").only("bookID", "name", "tag", "des", "title")
    if not all_books:
        return {"error": "ไม่พบข้อมูลนิยาย"}

    vectors = []
    metas = []
    for b in all_books:
        tag_text = b.tag.replace(",", " ") if b.tag else ""
        name_text = b.name or ""
        title_text = b.title or ""
        combined_text = f"{name_text} {tag_text} {title_text}"
        vec = get_vector(combined_text)
        vectors.append(vec)
        metas.append({
            "bookID": b.bookID,
            "name": name_text,
            "tag": tag_text,
            "title": title_text
        })

    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    total_updated = 0
    for i, base in enumerate(metas):
        bookID = base["bookID"]
        query_vec = vectors[i].reshape(1, -1)
        D, I = index.search(query_vec, topn + 1)  # +1 เผื่อมีตัวเอง

        recommendations = []
        for idx, score in zip(I[0], D[0]):
            if idx == i:
                continue  # ข้ามตัวเอง
            sim = metas[idx]
            recommendations.append({
                "bookID": sim["bookID"],
                "name": sim["name"],
                "tag": sim["tag"],
                "title": sim["title"],
                "score": float(score),
                "rank": len(recommendations) + 1
            })
            if len(recommendations) == topn:
                break

        # 🔹 Clear old recommendations for this book
        await RecommendSimilar.filter(bookID=bookID).delete()

        # 🔹 Create new recommendations in bulk
        reco_objs = [
            RecommendSimilar(
                bookID=bookID,
                similarID=rec["bookID"],
                score=rec["score"],
                rank=rec["rank"],
                updated_at=datetime.now()
            )
            for rec in recommendations
        ]
        await RecommendSimilar.bulk_create(reco_objs)
        total_updated += 1
        print(f"✅ อัปเดต: {bookID} จำนวน {len(recommendations)} รายการ")

    return {
        "status": "ok",
        "total_books": len(all_books),
        "updated": total_updated
    }
                    
        

@recommend_router.get("/recommend/by-bookID")
async def recommend_by_bookID(bookID: str = Query(...), topn: int = 10):
    base = await BookTran.filter(bookID=bookID, status="publish").only("bookID", "name", "tag", "des", "title").first()
    if not base:
        return {"error": "ไม่พบข้อมูลนิยาย"}
    
    tag_text = base.tag.replace(",", " ") if base.tag else ""
    name_text = base.name or ""
    title_text = base.title or ""
    combined_text = f"{name_text} {tag_text} {title_text}"
    query_vec = np.array(get_vector(combined_text), dtype=np.float32).reshape(1, -1)

    vectors, meta = await load_all_books_and_vectors([bookID])
    if vectors.shape[0] == 0:
        return {"error": "ไม่พบนิยายอื่นเพื่อเปรียบเทียบ"}

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(query_vec)
    index.add(vectors)
    D, I = index.search(query_vec, topn)

    recommendations = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        b = meta[idx]
        b["score"] = float(score)
        b["rank"] = rank
        recommendations.append(b)

        # ⬇️ Check + insert or update
        existing = await RecommendSimilar.filter(bookID=bookID, similarID=b["bookID"]).first()
        if existing:
            await existing.delete()  # ⬅️ ลบข้อมูลเก่าออกก่อน
            print(f"🗑 ลบรายการเดิม: {bookID} → {b['bookID']}")

        # ไม่ว่าเจอหรือไม่ ก็สร้างใหม่เสมอ
        try:
            await RecommendSimilar.create(
                bookID=bookID,
                similarID=b["bookID"],
                score=b["score"],
                rank=rank,
                updated_at=datetime.now()
            )
            print(f"✅ สร้างใหม่: {bookID} → {b['bookID']} (rank {rank})")
        except IntegrityError:
            print(f"⚠️ duplicate recommend for {bookID} - {b['bookID']}")

    return {
        "base_book": {
            "bookID": base.bookID,
            "name": base.name,
            "tag": base.tag,
            "title": base.title,
        },
        "recommendations": recommendations
    }

@recommend_router.get("/recommend/random-similar")
async def recommend_random_similar(q: str = Query(...), topn: int = 30, pick: int = 5):
    df = await load_book_df()
    query_vec = get_vector(q)
    df["score"] = df["vector"].apply(lambda v: cosine_similarity([query_vec], [v])[0][0])
    similar_df = df.sort_values("score", ascending=False).head(topn)
    result = similar_df.sample(n=min(pick, len(similar_df)))
    return result.sort_values("score", ascending=False)[["bookID", "name", "tag", "score"]].to_dict(orient="records")

@recommend_router.get("/user-hit-read")
async def user_hit_read(user_id: str):
    hits = await UserHitRead.filter(user_id=user_id).prefetch_related("bt", "bte")
    tags = set()
    bookIds = set()
    for hit in hits:
        if hit.bt:
            if hit.bt.tag:
                tags.update(tag.strip() for tag in hit.bt.tag.split(",") if tag.strip())
            bookIds.add(hit.bt.bookID)

    if not tags:
        return []

    df = await load_book_df_with_not_in(bookIds)
    if df.empty:
        return []
    query_vec = get_vector(" ".join(tags))
    df["score"] = df["vector"].apply(lambda v: cosine_similarity([query_vec], [v])[0][0])
    result = df.sort_values("score", ascending=False).head(10)
    return {"books": result[["bookID", "name", "tag", "des", "score"]].to_dict(orient="records")}

class UserHitRequest(BaseModel):
    user_ids: List[str]

@recommend_router.post("/user-hit-read-multi")
async def user_hit_read_multi(request: UserHitRequest = Body(...)):
    result_by_user = {}
    for user_id in request.user_ids:
        hits = await UserHitRead.filter(user_id=user_id).prefetch_related("bt", "bte")
        tags = set()
        bookIds = set()
        for hit in hits:
            if hit.bt:
                if hit.bt.tag:
                    tags.update(tag.strip() for tag in hit.bt.tag.split(",") if tag.strip())
                bookIds.add(hit.bt.bookID)

        if not tags:
            result_by_user[user_id] = []
            continue
        df = await load_book_df_with_not_in(bookIds)
        if df.empty:
            result_by_user[user_id] = []
            continue

        query_vec = get_vector(" ".join(tags))
        df["score"] = df["vector"].apply(lambda v: cosine_similarity([query_vec], [v])[0][0])
        top_df = df.sort_values("score", ascending=False).head(10)
        result_by_user[user_id] = top_df[["bookID", "name", "tag", "des", "score"]].to_dict(orient="records")

    return JSONResponse(content=result_by_user)