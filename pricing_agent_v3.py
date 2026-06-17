"""
AI Үнийн Санал Болгох Агент  v3
Монголын жижиглэн худалдааны үнийг оновчлогч систем.

Бүтээгдэхүүн тус бүрийн борлуулалтын түүх, эрэлтийн хэв маяг,
ашиг болон өртгийг шинжлэн оновчтой үнийг санал болгоно.

Гаралт: JSON + CSV + HTML Dashboard (гурвыг нэгэн зэрэг)

Хэрэглэх:
    python pricing_agent.py --csv int.csv
    python pricing_agent.py --csv int.csv --api-key ТАНЫ_ТҮЛХҮҮР --top 200
"""

import json
import os
import sys
import argparse
from datetime import datetime

import pandas as pd

try:
    from google import genai
    from google.genai import types
    GEMINI_БОЛОМЖТОЙ = True
except ImportError:
    GEMINI_БОЛОМЖТОЙ = False


# ─────────────────────────────────────────────────────────────────────────────
# Өгөгдөл Ачаалах ба Шинжлэх
# ─────────────────────────────────────────────────────────────────────────────

def ачаалах_шинжлэх(csv_зам: str) -> pd.DataFrame:
    """
    Борлуулалтын CSV файлыг ачаалж, буруу/хог мөрүүдийг цэвэрлэн,
    бүтээгдэхүүн тус бүрийн нэгтгэсэн үзүүлэлтүүдийг тооцоолно.
    """
    print(f"\n📂 CSV уншиж байна: {csv_зам}")
    df = pd.read_csv(csv_зам)
    print(f"   ✅ {len(df):,} мөр ачааллаа")

    # ── Өгөгдлийн чанарын шүүлтүүр ──────────────────────────────────────────
    анхны_тоо = len(df)

    # Заавал байх баганууд хоосон байвал хасах
    df = df.dropna(subset=["Price", "BasePrice", "Profit", "SalesQty", "Amount"])

    # Тэг буюу сөрөг үнэ хасах
    df = df[df["Price"] > 0].copy()
    df = df[df["BasePrice"] > 0].copy()
    df = df[df["SalesQty"] > 0].copy()

    # Мэдэгдэхүйц буруу үнэ шүүх (1₮, 2₮ гэх мэт өгөгдлийн алдаа)
    үнийн_медиан = df["Price"].median()
    df = df[df["Price"] >= 50].copy()                          # 50₮-аас бага = алдаа
    df = df[df["BasePrice"] >= 50].copy()                      # суурь үнэ мөн адил
    df = df[df["BasePrice"] < үнийн_медиан * 500].copy()      # хэт өндөр суурь үнэ = алдаа

    хасагдсан = анхны_тоо - len(df)
    print(f"   🧹 {хасагдсан:,} буруу/хог мөр хасагдсан → {len(df):,} мөр үлдсэн")

    # Огноог задлах
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    df["ЖилСар"]   = df["SaleDate"].dt.to_period("M")

    # ── Бүтээгдэхүүн тус бүрээр нэгтгэх ─────────────────────────────────────
    print("\n📊 Бүтээгдэхүүн тус бүрийн үзүүлэлт тооцоолж байна...")
    нэгтгэсэн = df.groupby(
        ["ItemPkId", "CategoryPkId", "GroupPkId", "BrandPkId"]
    ).agg(
        нийт_тоо           = ("SalesQty",  "sum"),
        нийт_орлого        = ("Amount",    "sum"),
        нийт_ашиг          = ("Profit",    "sum"),
        дундаж_үнэ         = ("Price",     "mean"),
        суурь_үнэ          = ("BasePrice", "mean"),
        хамгийн_бага_үнэ   = ("Price",     "min"),
        хамгийн_өндөр_үнэ  = ("Price",     "max"),
        идэвхтэй_сар       = ("ЖилСар",   lambda x: x.nunique()),
        сүүлд_зарсан       = ("SaleDate",  "max"),
        эхлэж_зарсан       = ("SaleDate",  "min"),
    ).reset_index()

    # ── Гаргаж авсан үзүүлэлтүүд ─────────────────────────────────────────────
    нэгтгэсэн["ашгийн_хувь"]         = (нэгтгэсэн["нийт_ашиг"] / нэгтгэсэн["нийт_орлого"] * 100).round(2)
    нэгтгэсэн["сарын_дундаж_тоо"]    = (нэгтгэсэн["нийт_тоо"] / нэгтгэсэн["идэвхтэй_сар"]).round(1)
    нэгтгэсэн["өртөг"]               = (нэгтгэсэн["дундаж_үнэ"] - (нэгтгэсэн["нийт_ашиг"] / нэгтгэсэн["нийт_тоо"])).round(0)
    нэгтгэсэн["хөнгөлөлтийн_хувь"]   = ((нэгтгэсэн["суурь_үнэ"] - нэгтгэсэн["дундаж_үнэ"]) / нэгтгэсэн["суурь_үнэ"] * 100).clip(lower=0).round(2)
    нэгтгэсэн["сарын_орлого"]        = (нэгтгэсэн["нийт_орлого"] / нэгтгэсэн["идэвхтэй_сар"]).round(0)
    нэгтгэсэн["сүүлд_зарснаас_хоног"] = (pd.Timestamp.now() - нэгтгэсэн["сүүлд_зарсан"]).dt.days

    # Эрэлтийн ангилал — датасетийн бодит тархалт дээр үндэслэн
    доод_босго  = нэгтгэсэн["сарын_дундаж_тоо"].quantile(0.25)
    өндөр_босго = нэгтгэсэн["сарын_дундаж_тоо"].quantile(0.75)

    def эрэлт_ангилах(q):
        if q >= өндөр_босго: return "Өндөр"
        elif q >= доод_босго: return "Дунд"
        else:                 return "Бага"

    def ашиг_ангилах(m):
        if m < 0:    return "Алдагдалтай"
        elif m < 20: return "Бага ашиг"
        elif m < 40: return "Дунд ашиг"
        else:        return "Сайн ашиг"

    нэгтгэсэн["эрэлтийн_түвшин"] = нэгтгэсэн["сарын_дундаж_тоо"].apply(эрэлт_ангилах)
    нэгтгэсэн["ашгийн_ангилал"]  = нэгтгэсэн["ашгийн_хувь"].apply(ашиг_ангилах)

    # Хязгааргүй ба хоосон утга цэвэрлэх
    нэгтгэсэн["ашгийн_хувь"] = нэгтгэсэн["ашгийн_хувь"].replace([float("inf"), float("-inf")], 0).fillna(0)
    нэгтгэсэн["өртөг"]       = нэгтгэсэн["өртөг"].replace([float("inf"), float("-inf")], 0).fillna(0)

    # Хураангуй
    print(f"   ✅ {len(нэгтгэсэн):,} бүтээгдэхүүн бэлэн боллоо")
    print(f"\n   📈 Ашгийн тархалт:")
    print(f"      🔴 Алдагдалтай : {(нэгтгэсэн['ашгийн_хувь'] < 0).sum():,}")
    print(f"      🟡 Бага ашиг   : {((нэгтгэсэн['ашгийн_хувь'] >= 0) & (нэгтгэсэн['ашгийн_хувь'] < 20)).sum():,}")
    print(f"      🟢 Сайн ашиг   : {(нэгтгэсэн['ашгийн_хувь'] >= 20).sum():,}")
    print(f"\n   📊 Эрэлтийн тархалт:")
    print(f"      Өндөр : {(нэгтгэсэн['эрэлтийн_түвшин'] == 'Өндөр').sum():,}")
    print(f"      Дунд  : {(нэгтгэсэн['эрэлтийн_түвшин'] == 'Дунд').sum():,}")
    print(f"      Бага  : {(нэгтгэсэн['эрэлтийн_түвшин'] == 'Бага').sum():,}")

    return нэгтгэсэн


# ─────────────────────────────────────────────────────────────────────────────
# Үнийн Логик
# ─────────────────────────────────────────────────────────────────────────────

def үнэ_санал_болгох(мөр: pd.Series) -> dict:
    """
    Нэг бүтээгдэхүүнд бизнесийн дүрмүүдийг хэрэглэн шинэ үнэ санал болгоно.
    Эрэлт + өртөг + ашиг + идэвхгүй байсан хугацаа — дөрвийг нэгэн зэрэг харгалзана.
    """
    одоогийн_үнэ   = мөр["дундаж_үнэ"]
    суурь_үнэ      = мөр["суурь_үнэ"]
    ашиг           = мөр["ашгийн_хувь"]
    эрэлт          = мөр["эрэлтийн_түвшин"]
    сарын_тоо      = мөр["сарын_дундаж_тоо"]
    хөнгөлөлт      = мөр["хөнгөлөлтийн_хувь"]
    идэвхгүй_хоног = мөр["сүүлд_зарснаас_хоног"]

    # Өртгийн доод хязгаар: одоогийн үнийн 40%-иас доош зарахгүй
    өртөг = max(мөр["өртөг"], одоогийн_үнэ * 0.40)

    санал    = одоогийн_үнэ
    шалтгаан = []
    үйлдэл   = "Өөрчлөхгүй"
    яаралтай = "Бага зэрэг"

    # ── Дүрэм 1: Алдагдалтай ─────────────────────────────────────────────────
    if ашиг < 0:
        хамгийн_бага = өртөг * 1.15
        санал    = min(max(хамгийн_бага, одоогийн_үнэ * 1.20), суурь_үнэ)
        үйлдэл   = "Үнэ нэмэх"
        яаралтай = "Яаралтай"
        шалтгаан.append(f"Алдагдалтай зарагдаж байна ({ашиг:.1f}%)")
        шалтгаан.append(f"Өртөгт суурилсан доод үнэ: {хамгийн_бага:,.0f}₮")

    # ── Дүрэм 2: Бага ашиг + өндөр эрэлт ────────────────────────────────────
    elif ашиг < 20 and эрэлт == "Өндөр":
        санал    = min(одоогийн_үнэ * 1.12, суурь_үнэ)
        үйлдэл   = "Үнэ нэмэх"
        яаралтай = "Яаралтай"
        шалтгаан.append(f"Эрэлт өндөр ({сарын_тоо:.0f} ш/сар), ашиг бага ({ашиг:.1f}%)")
        шалтгаан.append("Эрэлтийг ашиглан ашгийг нэмэгдүүлэх боломж байна")

    # ── Дүрэм 3: Сайн ашиг + маш өндөр эрэлт ────────────────────────────────
    elif ашиг >= 40 and эрэлт == "Өндөр" and сарын_тоо >= 50:
        санал    = min(одоогийн_үнэ * 1.05, суурь_үнэ * 1.02)
        үйлдэл   = "Үнэ нэмэх"
        яаралтай = "Дунд"
        шалтгаан.append(f"Эрэлт маш өндөр ({сарын_тоо:.0f} ш/сар), ашиг сайн ({ашиг:.1f}%)")
        шалтгаан.append("Борлуулалт алдахгүйгээр аажим үнэ нэмэх боломж байна")

    # ── Дүрэм 4: Бага эрэлт + их хөнгөлөлттэй ч зарагдахгүй ────────────────
    elif эрэлт == "Бага" and хөнгөлөлт > 15:
        санал    = одоогийн_үнэ * 0.90
        үйлдэл   = "Үнэ бууруулах"
        яаралтай = "Дунд"
        шалтгаан.append(f"Эрэлт бага ({сарын_тоо:.1f} ш/сар), {хөнгөлөлт:.1f}% хөнгөлөлттэй ч зарагдахгүй")
        шалтгаан.append("Үнийг бууруулж борлуулалтыг нэмэгдүүлэх")

    # ── Дүрэм 5: Бага эрэлт + сайн ашиг ─────────────────────────────────────
    elif эрэлт == "Бага" and ашиг > 30 and хөнгөлөлт < 5:
        санал    = одоогийн_үнэ * 0.93
        үйлдэл   = "Үнэ бууруулах"
        яаралтай = "Дунд"
        шалтгаан.append(f"Эрэлт бага ({сарын_тоо:.1f} ш/сар), ашиг бууруулахыг зөвшөөрнө ({ашиг:.1f}%)")
        шалтгаан.append("Үнийн бага бууралт илүү олон худалдан авагч татна")

    # ── Дүрэм 6: Удаан хугацаанд зарагдаагүй ────────────────────────────────
    elif идэвхгүй_хоног > 60 and ашиг > 20:
        санал    = одоогийн_үнэ * 0.88
        үйлдэл   = "Үнэ бууруулах"
        яаралтай = "Дунд"
        шалтгаан.append(f"{идэвхгүй_хоног} хоногийн өмнө сүүлд зарагдсан")
        шалтгаан.append("Борлуулалт идэвхжүүлэхийн тулд үнэ бууруулах")

    # ── Дүрэм 7: Суурь үнээс шаардлагагүй доош ──────────────────────────────
    elif одоогийн_үнэ < суурь_үнэ * 0.95 and ашиг > 45 and эрэлт != "Бага":
        санал    = суурь_үнэ * 0.98
        үйлдэл   = "Үнэ нэмэх"
        яаралтай = "Дунд"
        шалтгаан.append(f"Суурь үнэ ({суурь_үнэ:,.0f}₮)-аас {хөнгөлөлт:.1f}% доош шаардлагагүй зарагдаж байна")
        шалтгаан.append("Ашиг ба эрэлт хангалттай — суурь үнэд ойртуулах боломжтой")

    # ── Дүрэм 8: Зохистой байна ──────────────────────────────────────────────
    else:
        шалтгаан.append(f"Одоогийн үнэ тохиромжтой: {ашиг:.1f}% ашиг, {эрэлт.lower()} эрэлт")

    # Өртгийн доод хязгаарын хамгаалалт
    if өртөг > 0 and санал < өртөг * 1.05:
        санал = өртөг * 1.10
        шалтгаан.append(f"⚠️ Өртгийн доод хязгаар хэрэгжлээ: {өртөг:,.0f}₮")

    # 10-ын үржвэрт дугуйлах
    санал = round(санал / 10) * 10

    өөрчлөлтийн_дүн  = санал - одоогийн_үнэ
    өөрчлөлтийн_хувь = (өөрчлөлтийн_дүн / одоогийн_үнэ * 100) if одоогийн_үнэ > 0 else 0

    # 0.5%-иас бага өөрчлөлт → өөрчлөхгүй
    if abs(өөрчлөлтийн_хувь) < 0.5:
        үйлдэл            = "Өөрчлөхгүй"
        яаралтай          = "Бага зэрэг"
        санал             = int(round(одоогийн_үнэ / 10) * 10)
        өөрчлөлтийн_дүн  = 0
        өөрчлөлтийн_хувь = 0.0

    return {
        "бараа_дугаар":        str(мөр["ItemPkId"]),
        "ангилал":             str(мөр["CategoryPkId"]),
        "бүлэг":               str(мөр["GroupPkId"]),
        "брэнд":               str(мөр["BrandPkId"]),
        "өртөг":               int(өртөг),
        "одоогийн_үнэ":        int(round(одоогийн_үнэ)),
        "суурь_үнэ":           int(round(суурь_үнэ)),
        "санал_болгох_үнэ":    int(санал),
        "өөрчлөлтийн_хувь":   round(өөрчлөлтийн_хувь, 1),
        "өөрчлөлтийн_дүн":    int(өөрчлөлтийн_дүн),
        "үйлдэл":              үйлдэл,
        "яаралтай":            яаралтай,
        "ашгийн_хувь":         round(ашиг, 1),
        "ашгийн_ангилал":      мөр["ашгийн_ангилал"],
        "эрэлтийн_түвшин":     эрэлт,
        "сарын_дундаж_тоо":    round(сарын_тоо, 1),
        "нийт_тоо":            int(мөр["нийт_тоо"]),
        "идэвхтэй_сар":        int(мөр["идэвхтэй_сар"]),
        "шалтгаан":            " | ".join(шалтгаан),
        "хөдөлгүүр":           "дүрэм-v3",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gemini AI Баяжуулалт (заавал биш)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_СИСТЕМИЙН_ЗААВАР = """Та Монголын жижиглэн худалдааны үнэ тогтоогч мэргэжилтэн.
Өгөгдөлд үндэслэн ЗӨВХӨН доорх JSON форматаар хариул. Монгол хэлээр бич.

{
  "шалтгаан_мн": "2-3 өгүүлбэр монгол тайлбар",
  "санал_болгох_үнэ": <тоо>
}"""

def gemini_тайлбар_нэмэх(санал: dict, api_түлхүүр: str) -> dict:
    """Дүрмийн саналд Gemini-гаар монгол тайлбар нэмнэ."""
    try:
        клиент = genai.Client(api_key=api_түлхүүр)
        prompt = f"""Бүтээгдэхүүний мэдээлэл:
- Одоогийн үнэ: {санал['одоогийн_үнэ']:,}₮
- Санал болгох үнэ: {санал['санал_болгох_үнэ']:,}₮
- Ашгийн хувь: {санал['ашгийн_хувь']}%
- Эрэлт: {санал['эрэлтийн_түвшин']} ({санал['сарын_дундаж_тоо']} ш/сар)
- Өртөг: {санал['өртөг']:,}₮
- Үйлдэл: {санал['үйлдэл']}
- Дүрмийн шалтгаан: {санал['шалтгаан']}
Монгол хэлээр тайлбар бич."""

        хариу = клиент.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_СИСТЕМИЙН_ЗААВАР,
                temperature=0.2,
                max_output_tokens=200,
            ),
        )
        текст = хариу.text.strip()
        if "```" in текст:
            текст = текст.split("```")[1]
            if текст.startswith("json"):
                текст = текст[4:]
        үр_дүн = json.loads(текст.strip())
        санал["шалтгаан_мн"] = үр_дүн.get("шалтгаан_мн", санал["шалтгаан"])
        санал["хөдөлгүүр"]   = "gemini"
    except Exception:
        санал["шалтгаан_мн"] = санал["шалтгаан"]
    return санал


# ─────────────────────────────────────────────────────────────────────────────
# CSV Гаралт
# ─────────────────────────────────────────────────────────────────────────────

def csv_хадгалах(үр_дүнгүүд: list, зам: str):
    """Үр дүнг CSV файлд хадгална — Excel-д нээхэд тохиромжтой."""
    df = pd.DataFrame(үр_дүнгүүд)

    # Багануудын дарааллыг тохируулах
    баганууд = [
        "бараа_дугаар", "ангилал", "бүлэг", "брэнд",
        "одоогийн_үнэ", "суурь_үнэ", "өртөг", "санал_болгох_үнэ",
        "өөрчлөлтийн_дүн", "өөрчлөлтийн_хувь",
        "үйлдэл", "яаралтай",
        "ашгийн_хувь", "ашгийн_ангилал",
        "эрэлтийн_түвшин", "сарын_дундаж_тоо",
        "нийт_тоо", "идэвхтэй_сар", "шалтгаан",
    ]
    df = df[[б for б in баганууд if б in df.columns]]
    df.to_csv(зам, index=False, encoding="utf-8-sig")  # utf-8-sig = Excel-д зөв харагдана
    print(f"   📊 CSV: {зам}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML Dashboard Гаралт
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_хадгалах(үр_дүнгүүд: list, хураангуй: dict, зам: str):
    """Шүүлтүүр, хайлттай HTML dashboard үүсгэнэ. Browser-т нээнэ."""

    нийт    = хураангуй["нийт_шинжилсэн"]
    нэмэх   = хураангуй["үнэ_нэмэх"]
    буурах  = хураангуй["үнэ_бууруулах"]
    хэвийн  = хураангуй["өөрчлөхгүй"]
    яаралтай = хураангуй["яаралтай_тоо"]

    # JSON өгөгдлийг JavaScript-д дамжуулах
    өгөгдөл_json = json.dumps(үр_дүнгүүд, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="mn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Үнийн Санал — Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #1a1a2e; }}

  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 28px 32px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
  .header .meta {{ font-size: 13px; opacity: 0.7; margin-top: 4px; }}
  .header .badge {{
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px; padding: 6px 16px; font-size: 13px;
  }}

  .stats {{ display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap; }}
  .stat-card {{
    background: white; border-radius: 12px; padding: 20px 24px;
    flex: 1; min-width: 160px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #e2e8f0;
  }}
  .stat-card.urgent   {{ border-left-color: #ef4444; }}
  .stat-card.increase {{ border-left-color: #f59e0b; }}
  .stat-card.decrease {{ border-left-color: #3b82f6; }}
  .stat-card.ok       {{ border-left-color: #10b981; }}
  .stat-card .num  {{ font-size: 32px; font-weight: 800; line-height: 1; }}
  .stat-card .lbl  {{ font-size: 13px; color: #64748b; margin-top: 6px; }}
  .stat-card.urgent .num   {{ color: #ef4444; }}
  .stat-card.increase .num {{ color: #f59e0b; }}
  .stat-card.decrease .num {{ color: #3b82f6; }}
  .stat-card.ok .num       {{ color: #10b981; }}

  .controls {{
    padding: 0 32px 16px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  }}
  .controls input {{
    padding: 10px 16px; border: 1px solid #d1d5db; border-radius: 8px;
    font-size: 14px; width: 280px; outline: none;
  }}
  .controls input:focus {{ border-color: #0f3460; box-shadow: 0 0 0 3px rgba(15,52,96,0.1); }}
  .controls select {{
    padding: 10px 14px; border: 1px solid #d1d5db; border-radius: 8px;
    font-size: 14px; background: white; outline: none; cursor: pointer;
  }}
  .controls select:focus {{ border-color: #0f3460; }}
  .count-label {{ font-size: 13px; color: #64748b; margin-left: auto; }}

  .table-wrap {{ padding: 0 32px 32px; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  thead {{ background: #1a1a2e; color: white; }}
  th {{ padding: 13px 14px; text-align: left; font-size: 12px;
        font-weight: 600; letter-spacing: 0.5px; white-space: nowrap; }}
  td {{ padding: 11px 14px; font-size: 13px; border-bottom: 1px solid #f1f5f9; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}

  .badge-urgency {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; white-space: nowrap;
  }}
  .u-яаралтай  {{ background: #fee2e2; color: #b91c1c; }}
  .u-дунд      {{ background: #fef3c7; color: #92400e; }}
  .u-бага      {{ background: #d1fae5; color: #065f46; }}

  .badge-action {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; white-space: nowrap;
  }}
  .a-нэмэх     {{ background: #fef3c7; color: #b45309; }}
  .a-буурах    {{ background: #dbeafe; color: #1d4ed8; }}
  .a-хэвийн   {{ background: #f1f5f9; color: #475569; }}

  .change-pos {{ color: #d97706; font-weight: 700; }}
  .change-neg {{ color: #2563eb; font-weight: 700; }}
  .change-zero {{ color: #94a3b8; }}

  .price {{ font-weight: 600; }}
  .reason-cell {{ max-width: 300px; font-size: 12px; color: #475569;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

  .no-results {{
    text-align: center; padding: 60px; color: #94a3b8; font-size: 15px;
  }}
  .footer {{
    text-align: center; padding: 20px; font-size: 12px; color: #94a3b8;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="h1">🤖 AI Үнийн Санал Болгох Агент</div>
    <div class="meta">Монголын жижиглэн худалдааны үнийн оновчлогч · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
  </div>
  <div class="badge">v3 · Дүрэм-суурилсан</div>
</div>

<div class="stats">
  <div class="stat-card urgent">
    <div class="num">{яаралтай}</div>
    <div class="lbl">🔴 Яаралтай өөрчлөлт</div>
  </div>
  <div class="stat-card increase">
    <div class="num">{нэмэх}</div>
    <div class="lbl">📈 Үнэ нэмэх санал</div>
  </div>
  <div class="stat-card decrease">
    <div class="num">{буурах}</div>
    <div class="lbl">📉 Үнэ бууруулах санал</div>
  </div>
  <div class="stat-card ok">
    <div class="num">{хэвийн}</div>
    <div class="lbl">✅ Өөрчлөхгүй</div>
  </div>
  <div class="stat-card">
    <div class="num">{нийт:,}</div>
    <div class="lbl">📦 Нийт бүтээгдэхүүн</div>
  </div>
</div>

<div class="controls">
  <input type="text" id="search" placeholder="🔍 Бараа хайх (ID, ангилал, брэнд...)" oninput="шүүх()">
  <select id="urgency-filter" onchange="шүүх()">
    <option value="">Яаралтай байдал (бүгд)</option>
    <option value="Яаралтай">🔴 Яаралтай</option>
    <option value="Дунд">🟡 Дунд</option>
    <option value="Бага зэрэг">🟢 Бага зэрэг</option>
  </select>
  <select id="action-filter" onchange="шүүх()">
    <option value="">Үйлдэл (бүгд)</option>
    <option value="Үнэ нэмэх">📈 Үнэ нэмэх</option>
    <option value="Үнэ бууруулах">📉 Үнэ бууруулах</option>
    <option value="Өөрчлөхгүй">✅ Өөрчлөхгүй</option>
  </select>
  <select id="demand-filter" onchange="шүүх()">
    <option value="">Эрэлт (бүгд)</option>
    <option value="Өндөр">Өндөр эрэлт</option>
    <option value="Дунд">Дунд эрэлт</option>
    <option value="Бага">Бага эрэлт</option>
  </select>
  <span class="count-label" id="count-label"></span>
</div>

<div class="table-wrap">
  <table id="main-table">
    <thead>
      <tr>
        <th>#</th>
        <th>Бараа ID</th>
        <th>Ангилал</th>
        <th>Брэнд</th>
        <th>Одоогийн үнэ</th>
        <th>Санал болгох үнэ</th>
        <th>Өөрчлөлт</th>
        <th>Үйлдэл</th>
        <th>Яаралтай</th>
        <th>Ашиг %</th>
        <th>Эрэлт</th>
        <th>Ш/сар</th>
        <th>Шалтгаан</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="no-results" id="no-results" style="display:none">Хайлтад тохирох бүтээгдэхүүн олдсонгүй.</div>
</div>

<div class="footer">AI Үнийн Санал Болгох Агент v3 · {datetime.now().strftime("%Y-%m-%d")} · Нийт {нийт:,} бүтээгдэхүүн шинжлэгдсэн</div>

<script>
const ӨГӨГДӨЛ = {өгөгдөл_json};

function форматУнэ(n) {{
  return n.toLocaleString('mn-MN') + '₮';
}}

function яаралтайКласс(v) {{
  if (v === 'Яаралтай') return 'u-яаралтай';
  if (v === 'Дунд')     return 'u-дунд';
  return 'u-бага';
}}

function үйлдэлКласс(v) {{
  if (v.includes('нэмэх'))    return 'a-нэмэх';
  if (v.includes('бууруулах')) return 'a-буурах';
  return 'a-хэвийн';
}}

function өөрчлөлтКласс(v) {{
  if (v > 0)  return 'change-pos';
  if (v < 0)  return 'change-neg';
  return 'change-zero';
}}

function мөрҮүсгэх(д, idx) {{
  const хувь = д.өөрчлөлтийн_хувь;
  const тэмдэг = хувь > 0 ? '+' : '';
  return `<tr>
    <td style="color:#94a3b8;font-size:12px">${{idx}}</td>
    <td><strong>${{д.бараа_дугаар}}</strong></td>
    <td style="color:#64748b">${{д.ангилал}}</td>
    <td style="color:#64748b">${{д.брэнд}}</td>
    <td class="price">${{форматУнэ(д.одоогийн_үнэ)}}</td>
    <td class="price" style="color:#0f3460">${{форматУнэ(д.санал_болгох_үнэ)}}</td>
    <td class="${{өөрчлөлтКласс(хувь)}}">${{тэмдэг}}${{хувь.toFixed(1)}}%</td>
    <td><span class="badge-action ${{үйлдэлКласс(д.үйлдэл)}}">${{д.үйлдэл}}</span></td>
    <td><span class="badge-urgency ${{яаралтайКласс(д.яаралтай)}}">${{д.яаралтай}}</span></td>
    <td>${{д.ашгийн_хувь.toFixed(1)}}%</td>
    <td>${{д.эрэлтийн_түвшин}}</td>
    <td>${{д.сарын_дундаж_тоо}}</td>
    <td class="reason-cell" title="${{д.шалтгаан}}">${{д.шалтгаан}}</td>
  </tr>`;
}}

function шүүх() {{
  const хайлт    = document.getElementById('search').value.toLowerCase();
  const яаралтай = document.getElementById('urgency-filter').value;
  const үйлдэл   = document.getElementById('action-filter').value;
  const эрэлт    = document.getElementById('demand-filter').value;

  const шүүгдсэн = ӨГӨГДӨЛ.filter(д => {{
    const тохирох_хайлт = !хайлт ||
      д.бараа_дугаар.toLowerCase().includes(хайлт) ||
      д.ангилал.toString().toLowerCase().includes(хайлт) ||
      д.брэнд.toString().toLowerCase().includes(хайлт) ||
      д.бүлэг.toString().toLowerCase().includes(хайлт);
    const тохирох_яаралтай = !яаралтай || д.яаралтай === яаралтай;
    const тохирох_үйлдэл   = !үйлдэл   || д.үйлдэл   === үйлдэл;
    const тохирох_эрэлт    = !эрэлт    || д.эрэлтийн_түвшин === эрэлт;
    return тохирох_хайлт && тохирох_яаралтай && тохирох_үйлдэл && тохирох_эрэлт;
  }});

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = шүүгдсэн.map((д, i) => мөрҮүсгэх(д, i + 1)).join('');

  document.getElementById('no-results').style.display = шүүгдсэн.length === 0 ? 'block' : 'none';
  document.getElementById('count-label').textContent = шүүгдсэн.length.toLocaleString() + ' бүтээгдэхүүн харагдаж байна';
}}

// Эхлэлд бүгдийг харуулах
шүүх();
</script>
</body>
</html>"""

    with open(зам, "w", encoding="utf-8") as ф:
        ф.write(html)
    print(f"   🌐 Dashboard: {зам}  (browser-т нээнэ үү)")


# ─────────────────────────────────────────────────────────────────────────────
# Үндсэн Агент
# ─────────────────────────────────────────────────────────────────────────────

def агент_ажиллуулах(
    csv_зам:      str,
    api_түлхүүр:  str = None,
    дээд_тоо:     int = None,
    гаралт_угтвар: str = "үр_дүн",
    gemini_дээд:  int = 50,
) -> dict:
    """
    Бүрэн дамжуулах хоолой:
    CSV ачаалах → цэвэрлэх → шинжлэх → үнэ санал болгох → JSON + CSV + Dashboard хадгалах.
    """
    print("\n" + "═" * 55)
    print("  AI ҮНИЙН САНАЛ БОЛГОХ АГЕНТ  v3")
    print("  Монголын Жижиглэн Худалдааны Үнийн Оновчлогч")
    print("═" * 55)

    бүтээгдэхүүнүүд = ачаалах_шинжлэх(csv_зам)

    # Тэргүүлэх бүтээгдэхүүнүүдийг эхэлж сонгох
    if дээд_тоо:
        бүтээгдэхүүнүүд["тэргүүлэх_оноо"] = (
            (бүтээгдэхүүнүүд["ашгийн_хувь"] < 0).astype(int)           * 4 +
            (бүтээгдэхүүнүүд["ашгийн_хувь"] < 20).astype(int)          * 2 +
            (бүтээгдэхүүнүүд["эрэлтийн_түвшин"] == "Бага").astype(int)
        )
        боловсруулах = бүтээгдэхүүнүүд.nlargest(дээд_тоо, "тэргүүлэх_оноо")
        print(f"\n🎯 Тэргүүлэх {дээд_тоо} бүтээгдэхүүн боловсруулагдана")
    else:
        боловсруулах = бүтээгдэхүүнүүд
        print(f"\n🎯 Бүх {len(бүтээгдэхүүнүүд):,} бүтээгдэхүүн боловсруулагдана")

    хөдөлгүүрийн_нэр = "Gemini AI + Дүрэм" if api_түлхүүр else "Дүрэм-суурилсан v3"
    print(f"🤖 Хөдөлгүүр: {хөдөлгүүрийн_нэр}")
    print("─" * 55)

    үр_дүнгүүд = []
    gemini_тоо  = 0

    яаралтай_тэмдэг = {"Яаралтай": "🔴", "Дунд": "🟡", "Бага зэрэг": "🟢"}

    for байрлал, (_, мөр) in enumerate(боловсруулах.iterrows(), start=1):
        санал = үнэ_санал_болгох(мөр)

        if api_түлхүүр and санал["яаралтай"] == "Яаралтай" and gemini_тоо < gemini_дээд:
            санал       = gemini_тайлбар_нэмэх(санал, api_түлхүүр)
            gemini_тоо += 1
        else:
            санал["шалтгаан_мн"] = санал["шалтгаан"]

        үр_дүнгүүд.append(санал)

        if байрлал <= 30 or санал["яаралтай"] == "Яаралтай":
            тэмдэг = яаралтай_тэмдэг.get(санал["яаралтай"], "⚪")
            print(
                f"  {байрлал:>4}. {мөр['ItemPkId']:<14} "
                f"{санал['одоогийн_үнэ']:>8,}₮ → "
                f"{санал['санал_болгох_үнэ']:>8,}₮  "
                f"{санал['өөрчлөлтийн_хувь']:>+6.1f}%  "
                f"{тэмдэг} {санал['үйлдэл']}"
            )

    if len(үр_дүнгүүд) > 30:
        print(f"       ... нийт {len(үр_дүнгүүд):,} бүтээгдэхүүн шинжлэгдсэн")

    хураангуй = {
        "нийт_шинжилсэн":   len(үр_дүнгүүд),
        "үнэ_нэмэх":        sum(1 for р in үр_дүнгүүд if "нэмэх"       in р["үйлдэл"]),
        "үнэ_бууруулах":    sum(1 for р in үр_дүнгүүд if "бууруулах"   in р["үйлдэл"]),
        "өөрчлөхгүй":       sum(1 for р in үр_дүнгүүд if "Өөрчлөхгүй" in р["үйлдэл"]),
        "яаралтай_тоо":     sum(1 for р in үр_дүнгүүд if р["яаралтай"] == "Яаралтай"),
        "дунд_тоо":         sum(1 for р in үр_дүнгүүд if р["яаралтай"] == "Дунд"),
        "бага_тоо":         sum(1 for р in үр_дүнгүүд if р["яаралтай"] == "Бага зэрэг"),
        "gemini_баяжуулсан": gemini_тоо,
    }

    # ── Гурван форматаар хадгалах ─────────────────────────────────────────────
    json_зам      = f"{гаралт_угтвар}.json"
    csv_зам_гар   = f"{гаралт_угтвар}.csv"
    dashboard_зам = f"{гаралт_угтвар}_dashboard.html"

    print(f"\n💾 Хадгалж байна...")

    # 1. JSON
    гаралт = {
        "үүсгэсэн_огноо":     datetime.now().isoformat(),
        "хөдөлгүүр":          хөдөлгүүрийн_нэр,
        "нийт_шинжилсэн_тоо": len(үр_дүнгүүд),
        "хураангуй":          хураангуй,
        "үр_дүнгүүд":         үр_дүнгүүд,
    }
    with open(json_зам, "w", encoding="utf-8") as ф:
        json.dump(гаралт, ф, ensure_ascii=False, indent=2)
    print(f"   📄 JSON: {json_зам}")

    # 2. CSV
    csv_хадгалах(үр_дүнгүүд, csv_зам_гар)

    # 3. HTML Dashboard
    dashboard_хадгалах(үр_дүнгүүд, хураангуй, dashboard_зам)

    # Эцсийн хураангуй
    print("\n" + "═" * 55)
    print("  ДҮГНЭЛТ")
    print("─" * 55)
    print(f"  Нийт шинжилсэн бүтээгдэхүүн : {len(үр_дүнгүүд):,}")
    print(f"  🔴 Яаралтай үнийн өөрчлөлт  : {хураангуй['яаралтай_тоо']:,}")
    print(f"  🟡 Дунд зэргийн тэргүүлэлт   : {хураангуй['дунд_тоо']:,}")
    print(f"  🟢 Өөрчлөлт шаардлагагүй     : {хураангуй['бага_тоо']:,}")
    print(f"  📈 Үнэ нэмэх санал            : {хураангуй['үнэ_нэмэх']:,}")
    print(f"  📉 Үнэ бууруулах санал        : {хураангуй['үнэ_бууруулах']:,}")
    if gemini_тоо:
        print(f"  🤖 Gemini тайлбар             : {gemini_тоо}")
    print(f"\n  📄 JSON     → {json_зам}")
    print(f"  📊 CSV      → {csv_зам_гар}")
    print(f"  🌐 Dashboard→ {dashboard_зам}")
    print("═" * 55 + "\n")

    return гаралт


# ─────────────────────────────────────────────────────────────────────────────
# Командын Мөрний Оролт
# ─────────────────────────────────────────────────────────────────────────────

def үндсэн():
    задлагч = argparse.ArgumentParser(
        description="AI Үнийн Санал Болгох Агент v3 — JSON + CSV + Dashboard"
    )
    задлагч.add_argument("--csv",        required=True,  help="Борлуулалтын CSV файлын зам")
    задлагч.add_argument("--api-key",    default=None,   help="Gemini API түлхүүр (заавал биш)")
    задлагч.add_argument("--top",        type=int,       default=None, help="Шинжлэх барааны тоо (өгөгдөөгүй бол бүгд)")
    задлагч.add_argument("--gemini-top", type=int,       default=50,   help="Gemini баяжуулах дээд тоо (өгөгдмөл: 50)")
    задлагч.add_argument("--output",     default="үр_дүн", help="Гаралтын файлын угтвар нэр (өгөгдмөл: үр_дүн)")

    аргументууд = задлагч.parse_args()

    if not os.path.exists(аргументууд.csv):
        print(f"❌ Файл олдсонгүй: {аргументууд.csv}")
        sys.exit(1)

    агент_ажиллуулах(
        csv_зам       = аргументууд.csv,
        api_түлхүүр   = аргументууд.api_key or os.environ.get("GEMINI_API_KEY"),
        дээд_тоо      = аргументууд.top,
        гаралт_угтвар = аргументууд.output,
        gemini_дээд   = аргументууд.gemini_top,
    )


if __name__ == "__main__":
    үндсэн()

    #python pricing_agent_v3.py --csv int.csv
    #open үр_дүн_dashboard.html