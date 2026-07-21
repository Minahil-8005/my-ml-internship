# ML-04 — Search Intelligence Data Contract

This document establishes the formal data contract for the Page Refresh / Re-scoring lane, verifying data grain, feature bucketing, null availability, and data leakage prevention.

---

## 1. Unit of analysis + time window

- **Unit of Analysis (Grain):** One row represents **one URL / Page (`page_url`)** aggregated over the observation period.
- **Time Window:** Mid-panel observation month: **`2026-03-01` to `2026-03-31`** (Treating June 2026 as a sealed test window).
- **Target Decision:** Predicting page traffic decay to select the top 50 URLs for monthly content refreshes.

---

## 2. Fields: feature / label / context / excluded

- **Feature Bucket (Knowable at Decision Moment):**
  - `clicks_past_30d`: Organic clicks accumulated over the preceding 30-day window.
  - `impressions_past_30d`: Total search visibility impressions in the preceding 30 days.
  - `ctr_past_30d`: Historical click-through rate calculated prior to decision time.
  - `position_mean_30d`: Mean SERP ranking position over the previous 30 days.
  - `days_since_last_update`: Total days elapsed since page content was last modified.

- **Label / Target Bucket:**
  - `is_declining` (Boolean): True if organic clicks dropped by >20% relative to the baseline period.

- **Context Bucket:**
  - `page_url`: Unique page resource identifier.
  - `category`: Structural content taxonomy classification.

- **Excluded Bucket & Reason:**
  - `future_clicks_next_30d` & `next_month_ctr`: **Excluded due to Data Leakage.** These metrics occur in the future relative to the decision timestamp.

---

## 3. Verify it with queries (grain, counts, missing values, windows)

```python
# Setup Hugging Face Token & Load Mid-panel Dataset (March 2026)
import os
import pandas as pd
import numpy as np
from datasets import load_dataset

# Authenticate via HF_TOKEN
hf_token = os.environ.get('HF_TOKEN')
dataset = load_dataset('FlyRank/internship-warehouse', split='train', token=hf_token)
df = pd.DataFrame(dataset)

# Filter for mid-panel month: 2026-03
df['date'] = pd.to_datetime(df['date'])
march_df = df[(df['date'] >= '2026-03-01') & (df['date'] <= '2026-03-31')].copy()

# Query 1: Verify Grain (1 Row = 1 Page URL)
page_counts = march_df.groupby('page_url').size()
duplicate_pages = (page_counts > 1).sum()
print(f"Query 1 [Grain Check]: Duplicate page_urls found = {duplicate_pages} (Expected 0)")

# Query 2: Row Count & Date Span Verification
min_date = march_df['date'].min().strftime('%Y-%m-%d')
max_date = march_df['date'].max().strftime('%Y-%m-%d')
total_rows = len(march_df)
print(f"Query 2 [Counts & Span]: Total Rows = {total_rows}, Date Range = {min_date} to {max_date}")

# Query 3: Availability Check using IS TRUE
surviving_rows = (march_df['has_gsc_data'] == True).sum() if 'has_gsc_data' in march_df.columns else march_df['clicks_past_30d'].notnull().sum()
print(f"Query 3 [Availability]: Surviving rows using IS TRUE filter = {surviving_rows} / {total_rows}")
