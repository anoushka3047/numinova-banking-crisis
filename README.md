# Numinova — Banking Crisis Detection & Risk Monitoring

**An intelligent banking risk analysis system designed to identify potential financial distress, detect suspicious activity, and monitor indicators of banking instability.**

## Overview

Financial institutions face several interconnected risks, including fraudulent activity, loan defaults, liquidity shortages, declining capital adequacy, and potential insolvency. Identifying these warning signs early can help institutions investigate potential problems and make better-informed decisions.

**Numinova** is a Python-based banking risk analysis project that combines machine learning and financial risk indicators to explore the detection of potential banking crises.

The project includes modules for fraud and financial risk detection, insolvency prediction, trust-related analysis, and banking risk visualization.

## Key Features

* **Financial Risk Detection:** Uses a machine learning classification model to assess potential bank failure risk.
* **Operational Risk Monitoring:** Evaluates indicators such as liquidity, capital adequacy, loan defaults, withdrawal rates, and customer sentiment.
* **Insolvency Prediction:** Includes a dedicated module for analyzing potential insolvency risk.
* **Trust Analysis:** Includes a module for exploring trust-related aspects of banking risk.
* **Banking Dashboard:** Provides a dashboard interface for presenting banking-related information and analysis.
* **Risk Dataset Analysis:** Uses structured CSV datasets containing financial risk indicators and prediction data.

## Technology Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model:** Random Forest Classifier in the fraud detection module
* **Visualization / Interface:** Python dashboard module
* **Data Storage:** CSV datasets

## Repository Structure

```text
numinova-banking-crisis/
│
├── bank_financial_risk_dataset_14000_10.csv
├── bank_risk_monitoring_dataset_1000plus (1).csv
├── fraud_detection.py
├── insolvency_prediction.py
├── insolvency_predictions_realistic.csv
├── nn.py
├── trust_building.py
└── user_banking_dashboard2.py
```

### File Descriptions

| File                                            | Purpose                                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `fraud_detection.py`                            | Trains and evaluates a Random Forest model and checks financial and operational risk indicators. |
| `insolvency_prediction.py`                      | Contains the insolvency prediction module.                                                       |
| `nn.py`                                         | Contains neural network-related code.                                                            |
| `trust_building.py`                             | Contains trust-related analysis functionality.                                                   |
| `user_banking_dashboard2.py`                    | Contains the banking dashboard implementation.                                                   |
| `bank_financial_risk_dataset_14000_10.csv`      | Financial risk dataset used by the fraud detection module.                                       |
| `bank_risk_monitoring_dataset_1000plus (1).csv` | Additional banking risk monitoring dataset.                                                      |
| `insolvency_predictions_realistic.csv`          | CSV file containing insolvency prediction data.                                                  |

## How It Works

The fraud detection module follows this workflow:

1. **Data loading:** Reads financial risk data from a CSV file.
2. **Data preparation:** Prepares the features and target labels for model training.
3. **Feature scaling:** Uses `StandardScaler` to scale input features.
4. **Model training:** Trains a Random Forest classifier.
5. **Model evaluation:** Calculates accuracy, cross-validation scores, a classification report, and a confusion matrix.
6. **Risk assessment:** Evaluates sample cases and generates warnings based on selected financial indicators.

The operational checks include liquidity ratio, capital adequacy ratio, loan default ratio, withdrawal rate, and customer sentiment score.

## Installation

### Prerequisites

* Python 3.10 or later recommended
* pip
* Git

### 1. Clone the repository

```bash
git clone https://github.com/anoushka3047/numinova-banking-crisis.git
cd numinova-banking-crisis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

For the fraud detection module:

```bash
pip install pandas numpy scikit-learn
```

Additional dependencies may be required for the dashboard, neural network, or other modules.

## Usage

Run the fraud detection module from the repository root:

```bash
python fraud_detection.py
```

The script trains the model, displays evaluation metrics, and evaluates predefined sample cases.

To explore the other components, inspect the corresponding Python files and install their required dependencies.

## Project Goals

Numinova aims to explore how machine learning and financial indicators can support:

* Early identification of potential financial distress.
* Monitoring of banking risk indicators.
* Analysis of potential insolvency.
* More accessible visualization of banking risk information.
* Further development of integrated banking risk monitoring tools.

## Limitations

* Model outputs depend on the quality, representativeness, and labeling of the available datasets.
* Synthetic or simulated data may not reflect real-world banking conditions.
* A high model accuracy does not necessarily mean reliable crisis detection.
* The system has not been established as a production-grade banking risk or fraud prevention solution.
* Predictions should be independently validated before being considered for operational decisions.

## Future Improvements

* Integrate explainable AI to identify the factors behind individual risk predictions.
* Add anomaly detection for unusual financial activity.
* Improve model validation and test for data leakage.
* Develop a unified dashboard combining fraud, insolvency, and operational risk indicators.
* Add automated alerts and historical risk trend analysis.
* Evaluate models on verified real-world or appropriately anonymized datasets.

## Disclaimer

Numinova is an educational and development project. Its outputs are not financial advice and should not be used as the sole basis for banking, lending, investment, or regulatory decisions.

## License

No license is specified in this repository yet. Add a `LICENSE` file if you intend to define how others may use, modify, and distribute the project.
