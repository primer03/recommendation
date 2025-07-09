from tortoise import fields, models, Tortoise
from enum import Enum

# 🔸 ENUMs
class YesNo(str, Enum):
    yes = "yes"
    no = "no"

class StatusEnum(str, Enum):
    publish = "publish"
    private = "private"
    delete = "delete"

class EndEnum(str, Enum):
    end = "end"
    not_end = "not_end"

class TypeEnum(str, Enum):
    novel = "novel"
    chat = "chat"

# 🔹 BookTran
class BookTran(models.Model):
    id = fields.IntField(pk=True)
    bookID = fields.CharField(max_length=255, unique=True)  # ⬅️ ต้อง unique เพื่อให้ FK ได้
    type = fields.CharEnumField(enum_type=TypeEnum, default=TypeEnum.novel)
    img = fields.CharField(max_length=255)
    name = fields.CharField(max_length=255)
    title = fields.TextField()
    des = fields.TextField()
    tag = fields.TextField()
    cat1: fields.ForeignKeyNullableRelation["BookCategory"] = fields.ForeignKeyField(
        "models.BookCategory", related_name="books_cat1", null=True, source_field="cat1"
    )
    cat2: fields.ForeignKeyNullableRelation["BookCategory"] = fields.ForeignKeyField(
        "models.BookCategory", related_name="books_cat2", null=True, source_field="cat2"
    )
    rate_img = fields.CharEnumField(enum_type=YesNo, default=YesNo.no)
    rate = fields.CharEnumField(enum_type=YesNo, default=YesNo.no)
    userID = fields.CharField(max_length=255)
    status = fields.CharEnumField(enum_type=StatusEnum, default=StatusEnum.publish)
    view = fields.IntField(default=0)
    end = fields.CharEnumField(enum_type=EndEnum, default=EndEnum.not_end)
    bgimg = fields.CharField(max_length=255, null=True)
    recommend = fields.CharEnumField(enum_type=YesNo, default=YesNo.no)
    noti_add = fields.CharEnumField(enum_type=YesNo, default=YesNo.yes)
    show_review = fields.CharEnumField(enum_type=YesNo, default=YesNo.yes)
    show_write = fields.CharEnumField(enum_type=YesNo, default=YesNo.yes)
    use_freecoin = fields.CharEnumField(enum_type=YesNo, default=YesNo.yes)
    fast_status = fields.CharEnumField(enum_type=YesNo, default=YesNo.no)
    createdAt = fields.DatetimeField(auto_now_add=True)
    updatedAt = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "book_tran"

# 🔹 BookTranEp
class BookTranEp(models.Model):
    id = fields.IntField(pk=True)
    epID = fields.CharField(max_length=255)
    groupID = fields.CharField(max_length=255, null=True)
    bookID = fields.CharField(max_length=255)

    # ✅ FK ไปยัง BookTran โดยอิง bookID (string)
    bt = fields.ForeignKeyField(
        "models.BookTran",
        related_name="episodes",
        to_field="bookID",
        source_field="bookID"
    )

    name = fields.CharField(max_length=255)
    coin = fields.IntField()
    discountID = fields.IntField(default=0)
    publishDay = fields.DatetimeField()
    status = fields.CharEnumField(enum_type=StatusEnum, default=StatusEnum.private)
    view = fields.IntField(default=0)
    order_by = fields.IntField(null=True)
    money_bag = fields.IntField(default=0)
    num_char = fields.IntField(default=0)
    num_word = fields.IntField(default=0)
    noti_add = fields.CharField(max_length=3, default="no")
    createdAt = fields.DatetimeField(auto_now_add=True)
    updatedAt = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "book_tran_ep"

# 🔹 UserHitRead
class UserHitRead(models.Model):
    id = fields.IntField(pk=True)
    user_id = fields.CharField(max_length=50)

    # ✅ FK ไป BookTran.id
    bt = fields.ForeignKeyField(
        "models.BookTran",
        related_name="hits",
        to_field="id",
        source_field="bt_id"
    )

    # ✅ FK ไป BookTranEp.id
    bte = fields.ForeignKeyField(
        "models.BookTranEp",
        related_name="hits",
        to_field="id",
        source_field="bte_id"
    )

    createdAt = fields.DatetimeField(auto_now_add=True)
    updatedAt = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_hit_read"

class BookCategory(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    order_by = fields.CharField(max_length=10, null=True)

    class Meta:
        table = "book_category"

class RecommendSimilar(models.Model):
    id = fields.IntField(pk=True)  # ✅ ใช้ id เป็น primary key
    bookID = fields.CharField(max_length=50)
    similarID = fields.CharField(max_length=50)
    score = fields.FloatField()
    rank = fields.IntField()
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "recommend_similar"
        unique_together = (("bookID", "similarID"),)

# DB Initializer
async def init_db():
    await Tortoise.init(
        db_url="mysql://dbreadevewrite:aXRALCBMRWNn8f2j@maindb.eveebook.com:3342/readeve_test",
        modules={"models": ["model"]}
    )
    
    # await Tortoise.generate_schemas()  # <- ใช้เฉพาะตอนพัฒนา หากต้องการให้ ORM สร้างตารางเอง
