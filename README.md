# Tourism Pressure in Europe (NUTS 3)

![Tourism pressure visualization](https://github.com/user-attachments/assets/2695b0a3-77a7-46be-9aab-9742ca39ac13)

This project analyzes tourism intensity across European regions using Eurostat data.

The main metric provides a proxy to understand where tourism may have a higher impact relative to the local population.

---

## 📊 What this project does

- Uses Eurostat datasets at **NUTS 3 level** (regional granularity)
- Cleans and integrates **tourism, population, and area** data
- Computes **tourism pressure indicators**
- Identifies regions with the **highest relative tourism intensity**
- Generates:
  - A **static chart (PNG)**
  - An **interactive dashboard (HTML)** with country selection

---

## 📁 Data sources

All data comes from **Eurostat**:

- Tourism nights → `tour_occ_nin2`
- Population → `demo_r_pjanaggr3`
- Area → `reg_area3`
- NUTS lookup table for region names

---
