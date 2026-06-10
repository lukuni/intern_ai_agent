"""
AI Үнийн Санал Болгох Агент v2
- Бүх барааг шинжилнэ
- Борлуулалт + эрэлт + ашиг + өртөг бүгдийг харгалзана
- Ахлагчийн шаардлагыг бүрэн хэрэгжүүлнэ
"""

import pandas as pd
import json
import argparse
import sys
import os
from datetime import datetime

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ════════════════════════════════════════════════════════
# 1. ӨГӨГДӨЛ ЦЭВЭРЛЭГЧ + ШИНЖИЛГЭЭ
# ════════════════════════════════════════════════════════

def load_and_analyze(csv_path: str) -> pd.DataFrame:
    print(f"\n📂 CSV уншиж байна: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✅ {len(df):,} мөр ачааллаа")

    # Цэвэрлэх
    before = len(df)
    df = df[df['Price'] > 0].copy()
    df = df.dropna(subset=['Price','BasePrice','Profit','SalesQty','Amount'])
    print(f"   🧹 {before - len(df):,} мөр цэвэрлэгдсэн → {len(df):,} мөр үлдсэн")

    # Огноо
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    df['YearMonth'] = df['SaleDate'].dt.to_period('M')

    # Бараа тус бүрээр нэгтгэх
    print("\n📊 Бараа тус бүрээр нэгтгэж байна...")
    agg = df.groupby(['ItemPkId','CategoryPkId','GroupPkId','BrandPkId']).agg(
        total_qty       = ('SalesQty',  'sum'),
        total_revenue   = ('Amount',    'sum'),
        total_profit    = ('Profit',    'sum'),
        avg_price       = ('Price',     'mean'),
        base_price      = ('BasePrice', 'mean'),
        min_price       = ('Price',     'min'),
        max_price       = ('Price',     'max'),
        months_active   = ('YearMonth', lambda x: x.nunique()),
        last_sale_date  = ('SaleDate',  'max'),
        first_sale_date = ('SaleDate',  'min'),
    ).reset_index()

    # Тооцоолол
    agg['margin_pct']       = (agg['total_profit'] / agg['total_revenue'] * 100).round(2)
    agg['avg_monthly_qty']  = (agg['total_qty'] / agg['months_active']).round(1)
    agg['cost_price']       = (agg['avg_price'] - (agg['total_profit'] / agg['total_qty'])).round(0)
    agg['discount_pct']     = ((agg['base_price'] - agg['avg_price']) / agg['base_price'] * 100).clip(lower=0).round(2)
    agg['revenue_per_month']= (agg['total_revenue'] / agg['months_active']).round(0)
    agg['days_since_last']  = (pd.Timestamp.now() - agg['last_sale_date']).dt.days

    # Эрэлтийн ангилал — таны CSV-н бодит тархалт дээр үндэслэн
    q25 = agg['avg_monthly_qty'].quantile(0.25)  # 2.0
    q75 = agg['avg_monthly_qty'].quantile(0.75)  # 24.5
    def demand_label(q):
        if q >= q75:   return 'Өндөр'
        elif q >= q25: return 'Дунд'
        else:          return 'Бага'
    agg['demand_level'] = agg['avg_monthly_qty'].apply(demand_label)

    # Ашгийн ангилал
    def margin_label(m):
        if m < 0:    return 'Алдагдалтай'
        elif m < 20: return 'Бага ашиг'
        elif m < 40: return 'Дунд ашиг'
        else:        return 'Сайн ашиг'
    agg['margin_label'] = agg['margin_pct'].apply(margin_label)

    # Inf/NaN цэвэрлэх
    agg['margin_pct'] = agg['margin_pct'].replace([float('inf'), float('-inf')], 0).fillna(0)
    agg['cost_price'] = agg['cost_price'].replace([float('inf'), float('-inf')], 0).fillna(0)

    print(f"   ✅ {len(agg):,} бараа бэлэн боллоо")
    print(f"\n   📈 Ашгийн тархалт:")
    print(f"      🔴 Алдагдалтай : {(agg['margin_pct'] < 0).sum():,} бараа")
    print(f"      🟡 Бага ашиг   : {((agg['margin_pct'] >= 0) & (agg['margin_pct'] < 20)).sum():,} бараа")
    print(f"      🟢 Сайн ашиг   : {(agg['margin_pct'] >= 20).sum():,} бараа")
    print(f"\n   📊 Эрэлтийн тархалт:")
    print(f"      Өндөр эрэлт : {(agg['demand_level'] == 'Өндөр').sum():,} бараа")
    print(f"      Дунд эрэлт  : {(agg['demand_level'] == 'Дунд').sum():,} бараа")
    print(f"      Бага эрэлт  : {(agg['demand_level'] == 'Бага').sum():,} бараа")

    return agg


# ════════════════════════════════════════════════════════
# 2. УХААЛАГ ҮНИЙН ЛОГИК
# ════════════════════════════════════════════════════════

def smart_pricing_rule(row) -> dict:
    """
    Ахлагчийн шаардлага дээр үндэслэсэн үнийн логик:
    борлуулалт + эрэлт + өртөг + ашиг → оновчтой үнэ
    """
    item_id    = row['ItemPkId']
    cur_price  = row['avg_price']
    base_price = row['base_price']
    cost_price = max(row['cost_price'], cur_price * 0.4)  # өртөг хамгийн багадаа 40%
    margin     = row['margin_pct']
    demand     = row['demand_level']
    monthly_q  = row['avg_monthly_qty']
    discount   = row['discount_pct']
    days_last  = row['days_since_last']

    suggested  = cur_price
    reasons    = []
    action     = 'Өөрчлөхгүй'
    urgency    = 'Хойшлуулж болно'

    # ── ТОХИОЛДОЛ 1: Алдагдалтай ──────────────────────────
    if margin < 0:
        # Өртөгөөс дээш 15% ашигтай болтол нэмэх
        min_viable = cost_price * 1.15
        suggested  = max(min_viable, cur_price * 1.20)
        suggested  = min(suggested, base_price)  # суурь үнээс хэтрэхгүй
        action     = 'Үнэ нэмэх'
        urgency    = 'Яаралтай'
        reasons.append(f"Алдагдалтай зарагдаж байна ({margin:.1f}%)")
        reasons.append(f"Өртөгийн тооцоогоор хамгийн бага үнэ: {min_viable:,.0f}₮")

    # ── ТОХИОЛДОЛ 2: Бага ашиг + Өндөр эрэлт ─────────────
    elif margin < 20 and demand == 'Өндөр':
        suggested = cur_price * 1.12
        suggested = min(suggested, base_price)
        action    = 'Үнэ нэмэх'
        urgency   = 'Яаралтай'
        reasons.append(f"Эрэлт өндөр ({monthly_q:.0f} ш/сар) боловч ашиг бага ({margin:.1f}%)")
        reasons.append("Өндөр эрэлтийг ашиглан ашгийг нэмэгдүүлэх боломжтой")

    # ── ТОХИОЛДОЛ 3: Сайн ашиг + Өндөр эрэлт ─────────────
    elif margin >= 40 and demand == 'Өндөр' and monthly_q >= 50:
        suggested = cur_price * 1.05
        suggested = min(suggested, base_price * 1.02)
        action    = 'Үнэ нэмэх'
        urgency   = 'Дунд'
        reasons.append(f"Эрэлт маш өндөр ({monthly_q:.0f} ш/сар), ашиг сайн ({margin:.1f}%)")
        reasons.append("Борлуулалт буурахгүйгээр үнэ аажим нэмэх боломж байна")

    # ── ТОХИОЛДОЛ 4: Бага эрэлт + Их хөнгөлөлт ──────────
    elif demand == 'Бага' and discount > 15:
        suggested = cur_price * 0.90
        action    = 'Үнэ бууруулах'
        urgency   = 'Дунд'
        reasons.append(f"Эрэлт бага ({monthly_q:.1f} ш/сар), {discount:.1f}% хөнгөлөлттэй ч зарагдахгүй")
        reasons.append("Үнийг бууруулж борлуулалтыг нэмэгдүүлэх")

    # ── ТОХИОЛДОЛ 5: Бага эрэлт + Сайн ашиг ─────────────
    elif demand == 'Бага' and margin > 30 and discount < 5:
        suggested = cur_price * 0.93
        action    = 'Үнэ бууруулах'
        urgency   = 'Дунд'
        reasons.append(f"Эрэлт бага ({monthly_q:.1f} ш/сар), ашиг хангалттай ({margin:.1f}%)")
        reasons.append("Үнийг бага зэрэг бууруулж эрэлтийг нэмэгдүүлэх")

    # ── ТОХИОЛДОЛ 6: Удаан зарагдаагүй ──────────────────
    elif days_last > 60 and margin > 20:
        suggested = cur_price * 0.88
        action    = 'Үнэ бууруулах'
        urgency   = 'Дунд'
        reasons.append(f"{days_last} хоногийн өмнө сүүлд зарагдсан")
        reasons.append("Борлуулалт идэвхжүүлэхийн тулд үнэ бууруулах")

    # ── ТОХИОЛДОЛ 7: Хөнгөлөлтгүй + Суурь үнээс доош ───
    elif cur_price < base_price * 0.95 and margin > 45 and demand != 'Бага':
        suggested = base_price * 0.98
        action    = 'Үнэ нэмэх'
        urgency   = 'Дунд'
        reasons.append(f"Суурь үнэ {base_price:,.0f}₮ боловч {discount:.1f}% хямдаар зарагдаж байна")
        reasons.append("Ашиг сайн, эрэлт байна — үнийг суурь үнэд ойртуулах")

    # ── ТОХИОЛДОЛ 8: Оновчтой байна ──────────────────────
    else:
        reasons.append(f"Одоогийн үнэ зохистой: ашиг {margin:.1f}%, эрэлт {demand.lower()}")

    # Үнэ өртөгөөс доош болохгүй
    if cost_price > 0 and suggested < cost_price * 1.05:
        suggested = cost_price * 1.10
        reasons.append(f"⚠️ Өртөгийн доод хязгаар хамгаалалт: {cost_price:,.0f}₮")

    suggested = round(suggested / 10) * 10  # 10-ын үржвэрт дугуйлах

    change_amt = suggested - cur_price
    change_pct = (change_amt / cur_price * 100) if cur_price > 0 else 0

    if abs(change_pct) < 0.5:
        action    = 'Өөрчлөхгүй'
        urgency   = 'Хойшлуулж болно'
        suggested = int(round(cur_price / 10) * 10)

    return {
        'item_id':         item_id,
        'category':        row['CategoryPkId'],
        'group':           row['GroupPkId'],
        'brand':           row['BrandPkId'],
        'cost_price':      int(cost_price),
        'current_price':   int(round(cur_price)),
        'base_price':      int(round(base_price)),
        'suggested_price': int(suggested),
        'change_pct':      f"{change_pct:+.1f}%",
        'change_amount':   int(change_amt),
        'action':          action,
        'urgency':         urgency,
        'margin_pct':      round(margin, 1),
        'margin_label':    row['margin_label'],
        'demand_level':    demand,
        'avg_monthly_qty': round(monthly_q, 1),
        'total_qty':       int(row['total_qty']),
        'months_active':   int(row['months_active']),
        'reason':          ' | '.join(reasons),
        'engine':          'rule-based-v2'
    }


# ════════════════════════════════════════════════════════
# 3. GEMINI API (AI тайлбар нэмэх)
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Та Монголын жижиглэн худалдааны үнэ тогтоогч мэргэжилтэн.
Өгөгдөлд үндэслэн ЗӨВХӨН доорх JSON форматаар хариул. Монгол хэлээр бич.

{
  "reason_mn": "2-3 өгүүлбэр монгол тайлбар",
  "suggested_price": тоо
}"""

def enrich_with_gemini(suggestion: dict, api_key: str) -> dict:
    """Дүрмийн саналд Gemini-гаар монгол тайлбар нэмэх."""
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""Барааны мэдээлэл:
- Одоогийн үнэ: {suggestion['current_price']:,}₮
- Санал болгох үнэ: {suggestion['suggested_price']:,}₮
- Ашгийн хувь: {suggestion['margin_pct']}%
- Эрэлт: {suggestion['demand_level']} ({suggestion['avg_monthly_qty']} ш/сар)
- Өртөг: {suggestion['cost_price']:,}₮
- Үйлдэл: {suggestion['action']}
- Шалтгаан: {suggestion['reason']}

Монгол хэлээр тайлбар бич."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=200,
            )
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        suggestion['reason_mn'] = result.get('reason_mn', suggestion['reason'])
        suggestion['engine']    = 'gemini'
    except Exception:
        suggestion['reason_mn'] = suggestion['reason']
    return suggestion


# ════════════════════════════════════════════════════════
# 4. ҮНДСЭН АГЕНТ
# ════════════════════════════════════════════════════════

def run_agent(csv_path, api_key=None, use_local=False,
              top_n=None, output_path="results.json", gemini_top=50):

    print("\n" + "═"*55)
    print("  AI ҮНИЙН САНАЛ БОЛГОХ АГЕНТ  v2")
    print("  Монголын жижиглэн худалдааны систем")
    print("═"*55)

    # Өгөгдөл ачаалах + шинжлэх
    items = load_and_analyze(csv_path)

    # Хэдэн бараа шинжлэх
    if top_n:
        # Тэргүүлэх барааг эхэлж — алдагдалтай → бага ашиг → бага эрэлт
        items['priority'] = (
            (items['margin_pct'] < 0).astype(int) * 4 +
            (items['margin_pct'] < 20).astype(int) * 2 +
            (items['demand_level'] == 'Бага').astype(int)
        )
        process_items = items.nlargest(top_n, 'priority')
        print(f"\n🎯 Тэргүүлэх {top_n} бараа сонгогдлоо")
    else:
        process_items = items
        print(f"\n🎯 Бүх {len(items):,} бараа шинжлэгдэнэ")

    engine_label = "Gemini API + Дүрэм" if api_key else "Дүрэм-суурилсан v2"
    print(f"🤖 AI хөдөлгүүр: {engine_label}")
    print("─"*55)

    results   = []
    gemini_used = 0

    for i, (_, row) in enumerate(process_items.iterrows(), 1):
        # Дүрэм дээр суурилсан санал
        suggestion = smart_pricing_rule(row)

        # Алдагдалтай эсвэл яаралтай барааг Gemini-гаар баяжуулах
        if api_key and suggestion['urgency'] == 'Яаралтай' and gemini_used < gemini_top:
            suggestion = enrich_with_gemini(suggestion, api_key)
            gemini_used += 1
        else:
            suggestion['reason_mn'] = suggestion['reason']

        results.append(suggestion)

        urgency_icon = {"Яаралтай": "🔴", "Дунд": "🟡",
                        "Хойшлуулж болно": "🟢"}.get(suggestion['urgency'], "⚪")
        if i <= 30 or suggestion['urgency'] == 'Яаралтай':
            print(f"  {i:>4}. {row['ItemPkId']:<14} "
                  f"{suggestion['current_price']:>8,}₮ → "
                  f"{suggestion['suggested_price']:>8,}₮  "
                  f"{suggestion['change_pct']:>7}  "
                  f"{urgency_icon} {suggestion['action']}")

    if len(results) > 30:
        print(f"       ... нийт {len(results):,} бараа шинжлэгдсэн")

    # Нэгтгэл
    summary = {
        'price_increase':  sum(1 for r in results if 'нэмэх'      in r['action']),
        'price_decrease':  sum(1 for r in results if 'бууруулах'   in r['action']),
        'no_change':       sum(1 for r in results if 'Өөрчлөхгүй' in r['action']),
        'urgent':          sum(1 for r in results if r['urgency']  == 'Яаралтай'),
        'medium':          sum(1 for r in results if r['urgency']  == 'Дунд'),
        'low':             sum(1 for r in results if r['urgency']  == 'Хойшлуулж болно'),
        'gemini_enriched': gemini_used,
        'avg_change_pct':  round(sum(r['change_amount'] for r in results) /
                                 max(sum(r['current_price'] for r in results), 1) * 100, 2),
    }

    output = {
        'generated_at':         datetime.now().isoformat(),
        'engine':               engine_label,
        'total_items_analyzed': len(results),
        'summary':              summary,
        'results':              results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "═"*55)
    print("  ДҮГНЭЛТ")
    print("─"*55)
    print(f"  Нийт шинжилсэн бараа  : {len(results):,}")
    print(f"  🔴 Үнэ нэмэх (яаралтай): {summary['urgent']:,}")
    print(f"  🟡 Үнэ өөрчлөх (дунд)  : {summary['medium']:,}")
    print(f"  🟢 Өөрчлөхгүй          : {summary['low']:,}")
    print(f"  📈 Үнэ нэмэх санал     : {summary['price_increase']:,}")
    print(f"  📉 Үнэ бууруулах санал : {summary['price_decrease']:,}")
    if gemini_used:
        print(f"  🤖 Gemini тайлбар      : {gemini_used} бараа")
    print(f"\n  💾 Хадгалсан: {output_path}")
    print("═"*55 + "\n")

    return output


# ════════════════════════════════════════════════════════
# 5. CLI
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Үнийн Санал Болгох Агент v2")
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--api-key",    default=None)
    parser.add_argument("--local",      action="store_true")
    parser.add_argument("--top",        type=int, default=None,
                        help="Шинжлэх барааны тоо (хоосон = бүгд)")
    parser.add_argument("--gemini-top", type=int, default=50,
                        help="Gemini-гаар баяжуулах барааны дээд тоо")
    parser.add_argument("--output",     default="results.json")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Файл олдсонгүй: {args.csv}")
        sys.exit(1)

    run_agent(
        csv_path   = args.csv,
        api_key    = args.api_key or os.environ.get("GEMINI_API_KEY"),
        use_local  = args.local,
        top_n      = args.top,
        output_path= args.output,
        gemini_top = args.gemini_top,
    )
