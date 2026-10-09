# Data

No data is included in this repository.

## Source

The course data is based on the public Kaggle dataset "Logistics and supply chain dataset" by DatasetEngineer:

https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset

- File on Kaggle: `dynamic_supply_chain_logistics_dataset.csv`
- License on Kaggle: CC0 (Public Domain)
- 26 columns, hourly records. The Kaggle description says the data comes from a logistics network in Southern California and spans January 2021 to January 2024. The course files run from 2021-01-01 00:00 to 2024-08-29 00:00, one row per hour (32,065 rows).

The match is based on the column names and their order, which are identical to the Kaggle file's, and on the dataset description. A row-by-row comparison was not done because the Kaggle download requires sign-in.

## File the notebook expects

`notebook/shipment_delay_prediction.ipynb` runs:

```python
pd.read_csv('supply_chain_data_part3_4.csv')
```

The path is relative, so place `supply_chain_data_part3_4.csv` in `notebook/` (the notebook's working directory) before running. `.gitignore` excludes CSV files.

## The course file is not the Kaggle file

`supply_chain_data_part3_4.csv` was provided by the course for Parts 3 and 4 of the project. It is a modified copy, so running the notebook on the Kaggle file will not reproduce the saved results.

Differences visible in the team's notebooks, compared with the earlier course file used in Part 2 (`supply_chain_dataset.csv`, which has the Kaggle schema):

- `handling_equipment_availability`, `order_fulfillment_status` and `cargo_condition_status` are 0/1 instead of continuous values between 0 and 1.
- In the first rows printed by both notebooks, `timestamp`, GPS coordinates, `traffic_congestion_level`, `loading_unloading_time`, `customs_clearance_time`, `driver_behavior_score` and `disruption_likelihood_score` match, while `fuel_consumption_rate`, `eta_variation_hours`, `warehouse_inventory_level`, `iot_temperature`, `route_risk_level`, `fatigue_monitoring_score`, `delay_probability` and `risk_classification` have different values.
- About 2% of values are missing in `iot_temperature`, `driver_behavior_score` and `fuel_consumption_rate`.
- The target `delivery_time_deviation` differs. In the Part 2 file it ranges from -2 to 10 hours (mean 5.18), and its correlation with every other numeric column is 0.013 or less in absolute value. In the Part 3/4 file it ranges from 0 to 45 hours (mean 7.49) and correlates with several features (for example 0.56 with `route_risk_level`).
