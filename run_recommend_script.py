import asyncio
import numpy as np
from datetime import datetime
import faiss
from tqdm import tqdm  # ⬅️ เพิ่มตรงนี้

from model import init_db, BookTran, RecommendSimilar
from vector import get_vector

async def run_recommend_all_books(topn: int = 10):
    await init_db()
    all_books = await BookTran.filter(status="publish").only("bookID", "name", "tag", "des", "title")
    if not all_books:
        print("❌ ไม่พบนิยาย")
        return

    vectors, metas = [], []
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
    for i, base in enumerate(tqdm(metas, desc="🔄 คำนวณนิยายคล้ายกัน", ncols=100)):
        bookID = base["bookID"]
        query_vec = vectors[i].reshape(1, -1)
        D, I = index.search(query_vec, topn + 1)

        recommendations = []
        for idx, score in zip(I[0], D[0]):
            if idx == i:
                continue
            sim = metas[idx]
            recommendations.append({
                "bookID": sim["bookID"],
                "score": float(score),
                "rank": len(recommendations) + 1
            })
            if len(recommendations) == topn:
                break

        await RecommendSimilar.filter(bookID=bookID).delete()
        objs = [
            RecommendSimilar(
                bookID=bookID,
                similarID=rec["bookID"],
                score=rec["score"],
                rank=rec["rank"],
                updated_at=datetime.now()
            )
            for rec in recommendations
        ]
        await RecommendSimilar.bulk_create(objs)
        total_updated += 1

    print(f"\n🎉 สำเร็จทั้งหมด {total_updated} เรื่อง")


if __name__ == "__main__":
    asyncio.run(run_recommend_all_books())
