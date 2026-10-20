import sqlite3

import pandas as pd
from sklearn.datasets import load_wine

# load the wine dataset into a SQLite database
wine = load_wine(as_frame=True)
df = wine.frame.rename(columns={"od280/od315_of_diluted_wines": "od280", "target": "cultivar"})

conn = sqlite3.connect("wine.db")
df.to_sql("wines", conn, if_exists="replace", index=False)

queries = {
    "Number of wines per cultivar": """
        SELECT cultivar, COUNT(*) AS wines
        FROM wines
        GROUP BY cultivar
    """,
    "Average alcohol, color and magnesium per cultivar": """
        SELECT cultivar,
               ROUND(AVG(alcohol), 2)         AS avg_alcohol,
               ROUND(AVG(color_intensity), 2) AS avg_color,
               ROUND(AVG(magnesium), 1)       AS avg_magnesium
        FROM wines
        GROUP BY cultivar
        ORDER BY avg_alcohol DESC
    """,
    "Top 5 wines by alcohol": """
        SELECT cultivar, alcohol, malic_acid, color_intensity
        FROM wines
        ORDER BY alcohol DESC
        LIMIT 5
    """,
    "Wines with alcohol above the overall average (subquery)": """
        SELECT cultivar, COUNT(*) AS above_avg
        FROM wines
        WHERE alcohol > (SELECT AVG(alcohol) FROM wines)
        GROUP BY cultivar
    """,
    "Cultivars with average proline over 700 (HAVING)": """
        SELECT cultivar, ROUND(AVG(proline)) AS avg_proline
        FROM wines
        GROUP BY cultivar
        HAVING AVG(proline) > 700
    """,
}

for title, sql in queries.items():
    print(f"\n--- {title} ---")
    print(pd.read_sql(sql, conn).to_string(index=False))

# export a summary for Excel / Power BI
summary = pd.read_sql("SELECT * FROM wines", conn)
summary.to_csv("wine_data.csv", index=False)
print("\nexported wine_data.csv")

conn.close()
