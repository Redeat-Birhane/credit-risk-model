# Credit Risk Model

This project is an end-to-end Credit Risk Modeling system that includes data processing, machine learning model training, inference, and a FastAPI-based deployment layer. It is designed to simulate a production-ready ML pipeline with CI/CD, Dockerization, and modular code structure.

---

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ └── eda.ipynb
├── src/
│ ├── data_processing.py
│ ├── train.py
│ ├── predict.py
│ └── api/
│ ├── main.py
│ └── pydantic_models.py
├── tests/
│ └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

---

## Project Goals

- Build a reproducible machine learning pipeline for credit risk prediction  
- Perform data cleaning, feature engineering, and model training  
- Expose model inference through a REST API using FastAPI  
- Ensure reproducibility using Docker  
- Maintain code quality with testing and CI/CD  

---

## Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Pytest
- Docker
- GitHub Actions

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/Redeat-Birhane/credit-risk-model.git
cd credit-risk-model