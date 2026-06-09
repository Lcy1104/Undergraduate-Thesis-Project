# init_city_adcode.py
import pandas as pd
import psycopg2
from pathlib import Path

# ========== 1. 一次性改成你要的值 ==========
PG_CFG = dict(
    host="localhost",
    port=5432,
    user="postgres",
    password="root",   # 你的密码
    dbname="kb_db"                  # 你的库名
)
EXCEL_PATH = Path(r"E:\system\Downloads\AMap_adcode_citycode.xlsx")   # 你的文件绝对路径
# ============================================

if not EXCEL_PATH.exists():
    print(f"❌ 文件不存在: {EXCEL_PATH}")
    exit()

conn = psycopg2.connect(**PG_CFG)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS city_adcode(
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) UNIQUE NOT NULL,
                adcode VARCHAR(10) NOT NULL,
                citycode VARCHAR(10));""")

df = pd.read_excel(EXCEL_PATH)
df = df[df['adcode'].notna()].copy()
df['city_name'] = df['中文名'].str.strip()
df['adcode']  = df['adcode'].astype(str).str.strip()
df['citycode']= df['citycode'].astype(str).replace(r'\\N','',regex=True).str.strip()

cnt = 0
for _,r in df.iterrows():
    cur.execute("""INSERT INTO city_adcode(city_name,adcode,citycode)
                   VALUES(%s,%s,%s)
                   ON CONFLICT(city_name) DO UPDATE
                   SET adcode=EXCLUDED.adcode,citycode=EXCLUDED.citycode;""",
                (r['city_name'],r['adcode'],r['citycode']))
    cnt += 1
conn.commit()
cur.close(); conn.close()
print(f"✅ 完成！共写入/更新 {cnt} 条城市编码。")