"""
REDACTED VERSION:
For security reasons and to prevent abuse, sensitive endpoints, headers, 
and specific target IDs have been redacted. The core logic for rate-limit 
handling, session management, and pagination remains intact.
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import random

list_url = "https://[HIDDEN_URL]"
session = requests.Session()
session.headers.update({
<<<<<<< HEAD
    "User-Agent": "[AGENT]",
=======
    "User-Agent": "[Agent]",
>>>>>>> 94a5ae72d4eb8ae908377d2c8b467e15a4586b1b
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://[REDACTED_DOMAIN]/",
    "Origin": "https://[REDACTED_DOMAIN]",
    "Connection": "keep-alive"
})
r = session.get(list_url)
soup = BeautifulSoup(r.text, "html.parser")

vendorCodes = ['Vendor_1', 'Vendor_2', 'Vendor_3', 'Vendor_4', 'Vendor_5']

output_file = "snappfood_comments.csv"
page_size = 100

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['code', 'customerId', 'rating', 'date', 'text', 'feeling'])
    for code in vendorCodes:
        print(f"starting {code}")
        page = 1
        while page <= 200:
            url = "[HIDDEN_URL]"
            params = {
                "vendorCode": code,
                "page": page,
                "page_size": page_size,
                "sortType": "[HIDDEN_SORT_PARAM]",
                "client": "[REDACTED_CLIENT_TYPE]"
            }

            for attempt in range(3):
                try:
                    r = session.get(url, params=params, timeout=10)
                    
                    if r.status_code == 429:
                        print("429! waiting...")
                        time.sleep(10 * (attempt + 1))
                        continue
                    
                    r.raise_for_status()
                    data = r.json()
                    break

                except Exception as e:
                    print("retrying...", e)
                    time.sleep(3)
            else:
                print("failed completely")
                break

            comments = data.get("data", {}).get("comments", [])
            if not comments:
                print(f"{code} ✅")
                break

            for c in comments:
                text = c.get("commentText", "").strip()
                rating = c.get("rate", 0)
                date = c.get("date")
                customerId = c.get("customerId")
                feeling = c.get('feeling')
                if text:
                    writer.writerow([code, customerId, text, rating, date, feeling])

            print(page)
            page += 1
            time.sleep(random.uniform(2,4))
