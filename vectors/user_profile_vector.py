import numpy as np
from vector import get_vector
from model import UserHitRead

async def get_user_profile_vector(user_id: str):
    hits = await UserHitRead.filter(user_id=user_id).prefetch_related("bt")

    vectors = []
    for hit in hits:
        book = hit.bt
        if book and book.status == "publish":
            tag = book.tag.replace(",", " ") if book.tag else ""
            name = book.name or ""
            title = book.title or ""
            combined = f"{name} {tag} {title}"
            vec = get_vector(combined)
            vectors.append(vec)

    if not vectors:
        return None  # ยังไม่มีพฤติกรรมเพียงพอ

    # ✅ เฉลี่ยเวกเตอร์ทั้งหมด
    profile_vec = np.mean(np.array(vectors, dtype=np.float32), axis=0)
    return profile_vec