# Advanced Anomaly Detection in NASA Logs

## Overview

This project applies anomaly detection techniques to NASA log data using Python and machine learning models. The project uses **Isolation Forest** to identify anomalies in preprocessed NASA log data. The logs are preprocessed to clean and vectorize textual data, normalize numeric fields, and handle missing values.

## Features

- **Preprocessing**: The NASA logs are preprocessed to handle missing values, convert date fields, and vectorize the `description` field using TF-IDF.
- **Anomaly Detection**: Isolation Forest is applied to detect anomalous log entries that deviate from the normal pattern of data.
- **Model Evaluation**: Anomalous entries are flagged and can be further analyzed for insights.

## Project Structure

```bash
.
├── README.md                     # Project documentation
├── main.py                       # Main Python script for anomaly detection
├── preprocessing.py              # Data preprocessing script (if needed separately)
├── preprocessed_nasa_logs.csv     # Preprocessed NASA log data
└── detected_anomalies.csv         # Output file with detected anomalies
```

## Example Output
Anomalies detected in the log data will be saved in detected_anomalies.csv. Each entry will be flagged as either normal (1) or anomalous (-1).

Example:

```
   year_issued  month_issued  description_length  anomaly
0         2017             1                567        -1
1         2016             5                345         1
2         2015            10                234        -1

```