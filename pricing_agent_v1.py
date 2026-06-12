"""
AI Үнийн Санал Болгох Агент
Монголын жижиглэн худалдааны байгууллагад зориулсан

Шаардлагатай суулгалт:
    pip install pandas google-generativeai

Ажиллуулах:
    python agent.py --csv your_data.csv --api-key YOUR_GEMINI_KEY
"""

import pandas as pd
import json
import argparse
import sys
import os
from datetime import datetime

# ── Gemini API ──────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Ollama (нөөц хөдөлгүүр) ────────────────────────────────────────────────
try:
    import urllib.request, json as _json
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# 1. ӨГӨГДӨЛ ЦЭВЭРЛЭГЧ
# ════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    CSV өгөгдлийг цэвэрлэж, агентэд тохиромжтой байдлаар бэлдэнэ.
    Returns: (cleaned_df, cleaning_report)
    """
    report = {}
    original_len = len(df)

    # 1. Тэг үнэтэй мөрүүдийг хасах
    zero_price = df[df['Price'] == 0]
    report['zero_price_removed'] = len(zero_price)
    df = df[df['Price'] > 0].copy()

    # 2. Хоосон утгуудыг арилгах
    null_rows = df.isnull().any(axis=1).sum()
    report['null_rows_removed'] = int(null_rows)
    df = df.dropna()

    # 3. Огноо форматлах
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])

    # 4. Ашгийн хувийг тооцоолох
    df['profit_margin_pct'] = (df['Profit'] / df['Amount'] * 100).round(2)
    df['profit_margin_pct'] = df['profit_margin_pct'].replace([float('inf'), float('-inf')], 0)

    # 5. Хөнгөлөлтийн хувийг тооцоолох
    df['discount_pct'] = ((df['BasePrice'] - df['Price']) / df['BasePrice'] * 100).round(2)
    df['discount_pct'] = df['discount_pct'].clip(lower=0)

    report['original_rows'] = original_len
    report['clean_rows'] = len(df)
    report['removed_total'] = original_len - len(df)

    return df, report


def aggregate_by_item(df: pd.DataFrame) -> pd.DataFrame:
    """
    Бараа тус бүрээр нэгтгэж агентэд илгээх өгөгдөл бэлдэнэ.
    """
    df['YearMonth'] = df['SaleDate'].dt.to_period('M')

    agg = df.groupby('ItemPkId').agg(
        category        = ('CategoryPkId',       'first'),
        group           = ('GroupPkId',           'first'),
        brand           = ('BrandPkId',           'first'),
        current_price   = ('Price',               'mean'),
        base_price      = ('BasePrice',           'mean'),
        total_revenue   = ('Amount',              'sum'),
        total_profit    = ('Profit',              'sum'),
        total_qty       = ('SalesQty',            'sum'),
        avg_margin_pct  = ('profit_margin_pct',   'mean'),
        avg_discount    = ('discount_pct',        'mean'),
        months_active   = ('YearMonth', lambda x: x.nunique()),
        last_sale       = ('SaleDate',            'max'),
    ).reset_index()

    agg['avg_monthly_qty']  = (agg['total_qty'] / agg['months_active']).round(1)
    agg['current_price']    = agg['current_price'].round(0)
    agg['base_price']       = agg['base_price'].round(0)
    agg['avg_margin_pct']   = agg['avg_margin_pct'].round(1)
    agg['avg_discount']     = agg['avg_discount'].round(1)

    # Ангилал: эрэлт өндөр / дунд / бага
    q33 = agg['avg_monthly_qty'].quantile(0.33)
    q66 = agg['avg_monthly_qty'].quantile(0.66)
    agg['demand_level'] = pd.cut(
        agg['avg_monthly_qty'],
        bins=[-1, q33, q66, float('inf')],
        labels=['Бага эрэлт', 'Дунд эрэлт', 'Өндөр эрэлт']
    )

    return agg


# ════════════════════════════════════════════════════════════════════════════
# 2. AI ХӨДӨЛГҮҮР
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Та Монголын жижиглэн худалдааны дэлгүүрийн
мэргэжлийн үнэ тогтоогч зөвлөх юм.

Өгөгдөлд үндэслэн ЗӨВХӨН доорх JSON форматаар хариул.
Тайлбарыг ЗӨВХӨН монгол хэлээр бич. Нэмэлт текст бүү нэм.

{
  "item_id": "...",
  "current_price": 00000,
  "suggested_price": 00000,
  "change_pct": "+X% эсвэл -X%",
  "action": "Үнэ нэмэх / Үнэ бууруулах / Өөрчлөхгүй",
  "reason": "Монгол хэлээр 1-2 өгүүлбэр тайлбар",
  "urgency": "Яаралтай / Дунд / Хойшлуулж болно"
}"""

ITEM_PROMPT_TEMPLATE = """Барааны мэдээлэл:
- ID: {item_id}
- Ангилал: {category} | Бүлэг: {group} | Брэнд: {brand}
- Одоогийн үнэ: {current_price:,.0f}₮
- Суурь үнэ: {base_price:,.0f}₮
- Сарын дундаж борлуулалт: {avg_monthly_qty} ширхэг
- Нийт борлуулалт: {total_qty:,.0f} ширхэг ({months_active} сар)
- Ашгийн хувь: {avg_margin_pct:.1f}%
- Хөнгөлөлт: {avg_discount:.1f}%
- Эрэлтийн түвшин: {demand_level}
- Нийт ашиг: {total_profit:,.0f}₮

Үнийн санал гарга."""


def call_gemini(item_data: dict, api_key: str) -> dict | None:
    """Gemini API-г ашиглан үнийн санал авах."""
    try:
        client = genai.Client(api_key=api_key)
        prompt = ITEM_PROMPT_TEMPLATE.format(**item_data)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=300,
            )
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return None


def call_ollama(item_data: dict, model: str = "qwen2.5") -> dict | None:
    """Ollama орон нутгийн загварыг ашиглан үнийн санал авах."""
    try:
        prompt = SYSTEM_PROMPT + "\n\n" + ITEM_PROMPT_TEMPLATE.format(**item_data)
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = _json.loads(resp.read())
            text = result.get("response", "").strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
    except Exception:
        return None


def get_suggestion(item_data: dict, api_key: str = None,
                   use_local: bool = False) -> dict | None:
    """
    3 давхаргат аюулгүй байдлын систем:
    Layer 1: Gemini API
    Layer 2: Ollama орон нутаг
    Layer 3: Дүрэм дээр суурилсан fallback
    """
    result = None

    if not use_local and api_key:
        result = call_gemini(item_data, api_key)
        if result:
            result['engine'] = 'gemini'
            return result

    # Layer 2: Ollama
    result = call_ollama(item_data)
    if result:
        result['engine'] = 'ollama'
        return result

    # Layer 3: Дүрэм дээр суурилсан fallback
    return rule_based_fallback(item_data)


def rule_based_fallback(item_data: dict) -> dict:
    """
    AI хөдөлгүүр байхгүй тохиолдолд дүрэм дээр суурилсан санал."""
    margin  = item_data['avg_margin_pct']
    demand  = str(item_data['demand_level'])
    current = item_data['current_price']

    if margin < 0:
        suggested = current * 1.15
        action    = "Үнэ нэмэх"
        reason    = "Алдагдалтай зарагдаж байна. Үнийг нэмэх шаардлагатай."
        urgency   = "Яаралтай"
        change    = "+15%"
    elif demand == 'Өндөр эрэлт' and margin > 50:
        suggested = current * 1.08
        action    = "Үнэ нэмэх"
        reason    = "Эрэлт өндөр, ашгийн хэмжээ сайн байна."
        urgency   = "Дунд"
        change    = "+8%"
    elif demand == 'Бага эрэлт' and margin > 30:
        suggested = current * 0.92
        action    = "Үнэ бууруулах"
        reason    = "Эрэлт бага байна. Үнийг бууруулж борлуулалтыг нэмэгдүүлэх."
        urgency   = "Дунд"
        change    = "-8%"
    else:
        suggested = current
        action    = "Өөрчлөхгүй"
        reason    = "Одоогийн үнэ зохистой байна."
        urgency   = "Хойшлуулж болно"
        change    = "0%"

    return {
        "item_id":         item_data['item_id'],
        "current_price":   int(current),
        "suggested_price": int(suggested),
        "change_pct":      change,
        "action":          action,
        "reason":          reason,
        "urgency":         urgency,
        "engine":          "rule-based"
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. ҮНДСЭН АГЕНТ
# ════════════════════════════════════════════════════════════════════════════

def run_agent(csv_path: str, api_key: str = None,
              use_local: bool = False, top_n: int = 50,
              output_path: str = "results.json"):
    """
    Үндсэн агентийн функц.
    """
    print("\n" + "═"*55)
    print("  AI ҮНИЙН САНАЛ БОЛГОХ АГЕНТ")
    print("  Монголын жижиглэн худалдааны систем")
    print("═"*55)

    # ── Өгөгдөл ачаалах ──────────────────────────────────────
    print(f"\n📂 CSV уншиж байна: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✅ {len(df):,} мөр ачааллаа")

    # ── Цэвэрлэх ────────────────────────────────────────────
    print("\n🧹 Өгөгдөл цэвэрлэж байна...")
    df, report = clean_data(df)
    print(f"   Тэг үнэтэй мөр хасагдсан : {report['zero_price_removed']:,}")
    print(f"   Хоосон утгатай мөр хасагдсан: {report['null_rows_removed']:,}")
    print(f"   Цэвэр өгөгдөл             : {report['clean_rows']:,} мөр")

    # ── Нэгтгэх ─────────────────────────────────────────────
    print("\n📊 Бараа тус бүрээр нэгтгэж байна...")
    items = aggregate_by_item(df)
    print(f"   ✅ {len(items):,} бараа бэлэн боллоо")

    # ── Тэргүүлэх барааг сонгох ─────────────────────────────
    # Алдагдалтай + өндөр эрэлттэй барааг эхэлж шинжлэх
    items['priority_score'] = (
        (items['avg_margin_pct'] < 0).astype(int) * 3 +
        (items['demand_level'] == 'Өндөр эрэлт').astype(int) * 2 +
        (items['avg_discount'] > 10).astype(int)
    )
    priority_items = items.nlargest(top_n, 'priority_score')

    print(f"\n🎯 Тэргүүлэх {top_n} бараа сонгогдлоо")
    print(f"   Алдагдалтай  : {(items['avg_margin_pct'] < 0).sum()} бараа")
    print(f"   Өндөр эрэлт  : {(items['demand_level'] == 'Өндөр эрэлт').sum()} бараа")

    engine_label = "Gemini API" if (api_key and not use_local) else \
                   "Ollama орон нутаг" if use_local else "Дүрэм-суурилсан"
    print(f"\n🤖 AI хөдөлгүүр: {engine_label}")
    print("─"*55)

    # ── Санал авах ────────────────────────────────────────────
    results = []
    for i, (_, row) in enumerate(priority_items.iterrows(), 1):
        item_data = {
            'item_id':        row['ItemPkId'],
            'category':       row['category'],
            'group':          row['group'],
            'brand':          row['brand'],
            'current_price':  row['current_price'],
            'base_price':     row['base_price'],
            'total_revenue':  row['total_revenue'],
            'total_profit':   row['total_profit'],
            'total_qty':      row['total_qty'],
            'avg_margin_pct': row['avg_margin_pct'],
            'avg_discount':   row['avg_discount'],
            'avg_monthly_qty':row['avg_monthly_qty'],
            'months_active':  row['months_active'],
            'demand_level':   row['demand_level'],
        }

        suggestion = get_suggestion(item_data, api_key, use_local)
        if suggestion:
            results.append(suggestion)
            urgency_icon = {"Яаралтай": "🔴", "Дунд": "🟡",
                           "Хойшлуулж болно": "🟢"}.get(
                suggestion.get('urgency',''), "⚪")
            print(f"  {i:>3}. {row['ItemPkId']:<15} "
                  f"{row['current_price']:>8,.0f}₮ → "
                  f"{suggestion.get('suggested_price',0):>8,.0f}₮  "
                  f"{suggestion.get('change_pct',''):>6}  "
                  f"{urgency_icon} {suggestion.get('action','')}")

    # ── Хадгалах ─────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "engine":       engine_label,
        "total_items_analyzed": len(results),
        "summary": {
            "price_increase": sum(1 for r in results if "нэмэх" in r.get('action','')),
            "price_decrease": sum(1 for r in results if "бууруулах" in r.get('action','')),
            "no_change":      sum(1 for r in results if "Өөрчлөхгүй" in r.get('action','')),
            "urgent":         sum(1 for r in results if r.get('urgency') == 'Яаралтай'),
        },
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Эцсийн тайлан ─────────────────────────────────────────
    print("\n" + "═"*55)
    print("  ДҮГНЭЛТ")
    print("─"*55)
    print(f"  Нийт шинжилсэн бараа : {len(results)}")
    print(f"  Үнэ нэмэх санал      : {output['summary']['price_increase']}")
    print(f"  Үнэ бууруулах санал  : {output['summary']['price_decrease']}")
    print(f"  Өөрчлөхгүй           : {output['summary']['no_change']}")
    print(f"  🔴 Яаралтай          : {output['summary']['urgent']}")
    print(f"\n  💾 Хадгалсан файл    : {output_path}")
    print("═"*55 + "\n")

    return output


# ════════════════════════════════════════════════════════════════════════════
# 4. CLI ИНТЕРФЭЙС
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Үнийн Санал Болгох Агент"
    )
    parser.add_argument("--csv",       required=True,  help="CSV файлын зам")
    parser.add_argument("--api-key",   default=None,   help="Gemini API түлхүүр")
    parser.add_argument("--local",     action="store_true", help="Ollama ашиглах")
    parser.add_argument("--top",       type=int, default=50, help="Шинжлэх барааны тоо")
    parser.add_argument("--output",    default="results.json", help="Гаралтын файл")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Файл олдсонгүй: {args.csv}")
        sys.exit(1)

    run_agent(
        csv_path    = args.csv,
        api_key     = args.api_key or os.environ.get("GEMINI_API_KEY"),
        use_local   = args.local,
        top_n       = args.top,
        output_path = args.output,
    )