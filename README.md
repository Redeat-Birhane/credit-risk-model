# Credit Risk Probability of Default (PD) Model

> A well-documented, interpretable credit scoring project built to Basel II standards, combining traditional statistical methods with modern alternative data approaches for robust default prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
  - [1. Basel II, Risk Measurement, and the Interpretability Imperative](#1-basel-ii-risk-measurement-and-the-interpretability-imperative)
  - [2. The Proxy Variable Problem: Why It Is Necessary and What Business Risks It Introduces](#2-the-proxy-variable-problem-why-it-is-necessary-and-what-business-risks-it-introduces)
  - [3. Model Trade-offs: Logistic Regression with WoE vs. Gradient Boosting in Regulated Finance](#3-model-trade-offs-logistic-regression-with-woe-vs-gradient-boosting-in-regulated-finance)
- [Data Sources & Features](#data-sources--features)
- [Methodology](#methodology)
- [Results & Evaluation](#results--evaluation)
- [References](#references)

---

## Project Overview

This project develops a **Credit Risk Probability of Default (PD) Model** grounded in the requirements of the Basel II Capital Accord. The objective is to predict the likelihood that a loan applicant will default, using both traditional financial indicators and alternative behavioral data derived from bank card transactions.

The model pipeline follows industry best practices: careful business understanding, defensible proxy-variable construction, Weight of Evidence (WoE) encoding, logistic regression as the primary interpretable model, and a thorough comparison with ensemble methods where performance gains justify complexity costs.

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
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Redeat-Birhane/credit-risk-model.git
cd credit-risk-model

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Credit Scoring Business Understanding

This section establishes the conceptual foundation necessary to make defensible, legally sound modeling choices. It draws on the Basel II framework, the HKMA's *Alternative Credit Scoring* white paper, the RFMS methodology (Huang, Zhou & Wang, 2018), and established credit risk literature from the Corporate Finance Institute and peer-reviewed research.

---

### 1. Basel II, Risk Measurement, and the Interpretability Imperative

#### Background: What Basel II Demands

The Basel II Capital Accord, introduced by the Basel Committee on Banking Supervision, fundamentally changed how banks calculate the minimum capital they must hold against credit losses. Rather than applying broad, undifferentiated risk weights, Basel II introduced the **Internal Ratings-Based (IRB) approach**, which allows banks to use their own empirically estimated risk parameters — most critically, the **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)** — to derive regulatory capital requirements. The central intent is to make capital reserves more accurately proportional to the actual credit risk being taken, creating both *risk sensitivity* (capital tracks real exposure) and *incentive compatibility* (better risk management reduces the capital burden).

This is not a permissive, trust-based arrangement. Banks may only use the IRB approach **subject to explicit approval from national supervisors** and after meeting rigorous minimum conditions around data quality, model validation, and documentation. The Basel Committee's own consultative documents identified the two primary barriers to using credit risk models for capital purposes as **data quality** and **the ability of banks and supervisors to validate model outputs**. A model a regulator cannot interrogate is a model a regulator will not approve.

#### Why Interpretability Becomes Non-Negotiable

The Basel II framework directly creates the demand for interpretable, well-documented models through three interlocking mechanisms:

**Supervisory auditability.** Under Pillar 2 (Supervisory Review), regulators must be able to examine a bank's internal rating system, understand the inputs, trace the logic, and challenge the outputs. As the HKMA's *Alternative Credit Scoring* white paper states explicitly: *"credit risk modelling in the banking sector requires that the model features be comprehensible to and interpretable by lending and risk officers."* A gradient-boosted ensemble that produces accurate predictions but cannot explain *why* a specific applicant was denied credit cannot survive a regulatory examination.

**Model validation requirements.** Basel II requires banks to independently validate their internal models. Validation means more than back-testing accuracy — it means explaining which variables are driving predictions, verifying that those drivers make economic sense, and demonstrating that the model is not arbitrarily penalizing protected characteristics through opaque feature interactions. The HKMA white paper directly identifies model interpretability as a systemic weakness of machine learning algorithms: *"an explanation of the relative contributions of the specific independent variables to the outcome of the machine learning model is hard to describe or prove."* Where a model cannot be explained, it cannot be validated to Basel standards.

**Accountability for lending decisions.** Fair lending laws in most jurisdictions — including adverse action notice requirements — oblige lenders to provide applicants with **specific, stated reasons** for credit denial. This is impossible with a pure black-box model. Logistic regression with WoE encoding satisfies this requirement naturally: each scorecard variable has a known, stable contribution to the final score, and the top reason codes can be extracted directly from the model output.

#### The Practical Implication for This Project

Basel II's emphasis on risk measurement does not merely suggest interpretability — it structurally requires it as a condition of regulatory approval. Every modeling choice in this project is therefore evaluated not just on predictive accuracy (AUC, Gini) but on whether it produces a model that a risk officer can explain, a validator can audit, and a regulator can approve. This is why the primary model is **Logistic Regression with Weight of Evidence (WoE) encoding** — a specification that has decades of proven regulatory acceptance — with ensemble models explored as supplementary benchmarks rather than production candidates unless explainability tools (SHAP, LIME) can adequately bridge the interpretability gap.

---

### 2. The Proxy Variable Problem: Why It Is Necessary and What Business Risks It Introduces

#### Why a Proxy Is Necessary

In an ideal credit scoring dataset, every observation carries a clean binary label: the borrower either defaulted or did not. In practice, particularly for **microcredit, MSME (Micro, Small and Medium-sized Enterprise) lending, and emerging-market consumer finance**, a direct default label is often unavailable or unreliable for one or more of the following reasons:

- **Loan book immaturity.** Many borrowers in a new or rapidly growing portfolio have not yet reached loan maturity, so their outcome (repaid vs. defaulted) is not yet observable.
- **Thin credit files.** Applicants who lack formal credit histories — the very population targeted by alternative credit scoring — have no past default events recorded in any bureau.
- **Survivorship bias.** As Huang, Zhou & Wang (2018) note in their RFMS study, a dataset of *approved* applicants excludes all rejected applicants, meaning the observed default rate does not reflect the true population risk distribution. This is a structural challenge affecting the entire microcredit industry.
- **Platform-specific data.** When transaction data is sourced from a payment platform rather than a formal credit bureau, the platform may record spending and transfer behaviors but not formal loan outcomes from other lenders.

In all these situations, a **proxy variable** — a measurable behavioral or financial signal that correlates reliably with the latent tendency to default — must stand in for the unobserved ground truth. Common proxy constructions include: whether a customer became 90+ days past due on any obligation within a defined observation window; whether a customer's account balance fell below a defined threshold before a scheduled repayment date; or behavioral deterioration signals derived from transaction frequency and monetary patterns, such as the RFMS framework (Recency, Frequency, Monetary value, and Standard deviation of transaction amounts) developed specifically for this purpose.

The HKMA white paper illustrates this in the context of e-merchant lending: *"A proxy parameter involves checking the refund history of the e-merchants"* — a behavioral signal that stands in for formal default because formal default data is unavailable for that population.

#### Business Risks Introduced by Proxy-Based Prediction

While proxies are a practical necessity, they introduce a set of specific, material business risks that must be actively managed:

**1. Label noise and miscalibration.** A proxy variable is, by definition, an imperfect approximation of the true outcome of interest. If the proxy is too conservative (e.g., flagging any 30-day delinquency as a proxy default), the model will be trained on a noisier, broader definition of default than the business actually cares about, leading to over-rejection of borderline-acceptable applicants and foregone revenue. If it is too permissive, the model will underestimate true risk.

**2. Temporal instability.** The behavioral patterns that predict proxy events during one economic regime (e.g., stable growth) may not predict actual defaults during a different regime (e.g., a credit crunch or pandemic). A model trained on proxy labels observed during a benign period may be dangerously miscalibrated when conditions change, without any obvious signal that the miscalibration has occurred.

**3. Proxy discrimination.** This is the most legally serious risk. A proxy variable that appears neutral on its surface may correlate with protected characteristics — race, gender, religion, national origin — in ways that violate fair lending laws. As industry practitioners highlight, even an apparently innocuous feature like geographic postal code can serve as a proxy for race if neighborhood demographics are correlated. A model trained to predict a behavioral proxy can learn and amplify these correlations without any explicit discriminatory intent. Every proxy used in this project must therefore be evaluated not only for predictive power but for **disparate impact** across protected classes before deployment.

**4. Selection bias and out-of-sample generalization.** Because the model is trained only on approved applicants (those for whom outcomes are observed), it will systematically have learned less about the risk profile of applicants who were historically rejected. When the approval policy changes or the model is deployed in a new market segment, this selection bias can cause the model to perform poorly on the new population — overconfidently granting credit to previously excluded applicants who carry higher actual risk.

**5. Regulatory challenge.** Regulators and internal model validators will scrutinize the proxy definition carefully. If the rationale for the proxy cannot be supported with economic reasoning and empirical evidence linking the proxy to actual default behavior, the model's approval may be delayed or denied.

---

### 3. Model Trade-offs: Logistic Regression with WoE vs. Gradient Boosting in a Regulated Financial Context

#### The Core Tension

The history of machine learning in credit risk is a story of a persistent tension between two desiderata that are difficult to satisfy simultaneously: **predictive performance** and **regulatory interpretability**. Ensemble methods — and gradient boosting in particular — consistently outperform logistic regression on raw discrimination metrics (AUC, Gini coefficient) across benchmark datasets. Research comparing machine learning algorithms for PD prediction consistently finds that ensemble methods such as XGBoost, CatBoost, and Random Forest outperform simpler models in handling complex patterns and imbalanced data. At the same time, the HKMA white paper and the Basel Committee's own guidance both make clear that black-box accuracy alone is insufficient for regulatory approval in a banking context.

The choice between these approaches is therefore not purely a technical one — it is a business and legal decision, with different trade-offs playing out across several dimensions.

#### Logistic Regression with Weight of Evidence (WoE)

**How it works.** In the WoE approach, continuous and categorical predictors are discretized into bins, and each bin is assigned a Weight of Evidence value that captures its log-odds relationship with the target variable. The transformed features are then used as inputs to a standard logistic regression. The final output is typically expressed as a **points-based scorecard**, where each variable contributes a known number of points to the total score.

**Advantages in a regulated context:**

- **Full transparency.** Every coefficient has a direct economic interpretation. A risk officer can look at the scorecard and explain precisely why applicant A received a score of 620 while applicant B received 540. This directly satisfies Basel Pillar 2 audit requirements and adverse action notice obligations.
- **Regulatory track record.** As the HKMA white paper notes, logistic regression *"is commonly used by both mission-driven lenders and financial lenders for binary objective situations"* and is *"intuitive, explicable, and faster than the other algorithms."* Regulators have decades of experience validating logistic scorecards; the approval pathway is well-established.
- **Stable, monotonic relationships.** WoE encoding enforces a monotonic relationship between each binned predictor and the outcome, which prevents the model from learning spurious non-linearities that may not generalize out-of-sample. This makes the model more robust across economic cycles.
- **Straightforward validation.** Population Stability Index (PSI), Characteristic Analysis, and Hosmer-Lemeshow goodness-of-fit tests — the standard regulatory validation toolkit — all apply naturally to logistic regression outputs.

**Disadvantages:**

- **Information loss from binning.** Discretizing continuous features into bins discards within-bin variation. The model cannot exploit subtle non-linear patterns that may genuinely improve prediction.
- **Limits on interaction terms.** Logistic regression does not automatically capture interactions between predictors (e.g., the combined effect of high transaction frequency *and* high transaction volatility). Feature engineering can partially address this, but it requires explicit domain knowledge and increases development time.
- **Performance ceiling.** On datasets with complex, non-linear risk signals — particularly alternative data like behavioral transaction sequences — logistic regression often has a materially lower AUC ceiling than ensemble methods. The RFMS study by Huang et al. demonstrated a 13.6% relative AUC improvement when moving from a basic score alone to a full feature set with logistic regression; ensemble methods on the same features would likely extract additional gains.

#### Gradient Boosting (XGBoost, LightGBM, CatBoost)

**How it works.** Gradient boosting builds an ensemble of shallow decision trees sequentially, each tree correcting the residual errors of the previous ones. The final prediction is a weighted sum of all trees' outputs. The model automatically learns complex non-linear relationships and feature interactions without requiring manual feature engineering.

**Advantages:**

- **Superior discrimination.** Across benchmark credit datasets, ensemble gradient boosting methods consistently achieve the highest AUC scores, particularly when the data contains non-linear risk signals, high-dimensional alternative features, or significant class imbalance. The HKMA's experimental results confirm this: *"Empirical observation indicates that Random Forest and XGBoost are more popular machine learning algorithms in recent years."*
- **Automatic feature interaction capture.** Gradient boosting discovers interaction effects without explicit specification, which is particularly valuable when working with the dense RFMS-style behavioral features (40+ variables across 10 spending categories) used in this project.
- **Handles missing data and high cardinality.** Modern implementations (LightGBM, CatBoost) handle missing values and categorical variables natively, reducing preprocessing burden.

**Disadvantages in a regulated context:**

- **Interpretability is a fundamental weakness.** The HKMA white paper is unambiguous: interpretability *"is a weakness of machine learning algorithms in general because an explanation of the relative contributions of the specific independent variables to the outcome of the machine learning model is hard to describe or prove."* Post-hoc tools like SHAP and LIME can provide feature-level explanations, but they approximate — they do not reproduce — the model's actual decision logic. A regulator who understands this distinction may not accept SHAP values as equivalent to a logistic scorecard for validation purposes.
- **Regulatory acceptance is not guaranteed.** As the HKMA notes, *"if a model is not highly interpretable, a bank may not be permitted to apply its insights to its business."* The use of gradient boosting in a production credit decision system requires additional model governance infrastructure (SHAP-based reason codes, stability monitoring, bias audits) that logistic regression does not.
- **Overfitting risk.** Gradient boosting models can overfit to training data, particularly with high-dimensional behavioral features. This requires careful cross-validation, hyperparameter tuning, and out-of-time testing — adding development complexity and validation overhead.
- **Proxy discrimination amplification.** Because gradient boosting captures complex non-linear interactions, it has a greater capacity to learn and amplify proxy discrimination signals than logistic regression. A logistic model's transparency makes discriminatory patterns easier to detect and correct; a gradient boosting model may encode the same discrimination across hundreds of trees in ways that are extremely difficult to audit.

#### Summary Comparison Table

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| **Predictive AUC** | Moderate — strong baseline | High — often 5–15% relative improvement |
| **Interpretability** | Full — direct coefficient → scorecard | Limited — requires SHAP/LIME approximations |
| **Regulatory acceptance** | Well-established, decades of precedent | Emerging — depends on jurisdiction and model governance |
| **Adverse action notices** | Native — scorecard reason codes | Requires post-hoc explanation tools |
| **Overfitting risk** | Low — WoE binning constrains complexity | Moderate–High — requires careful tuning |
| **Proxy discrimination risk** | Lower — transparent, auditable | Higher — complex interactions can encode bias opaquely |
| **Development complexity** | Moderate — WoE binning + validation | High — tuning, SHAP infrastructure, ongoing monitoring |
| **Basel II/IRB compatibility** | High — validated methodology | Conditional — requires additional governance controls |

#### The Decision Framework for This Project

Given the regulatory context, this project adopts the following strategy: **Logistic Regression with WoE encoding is the primary production model**. It is interpretable, auditable, and directly produces the scorecard format required for regulatory submission and adverse action compliance. Gradient Boosting models (XGBoost, LightGBM) are developed in parallel as **benchmark comparators**. If a gradient boosting model achieves a materially superior AUC (defined as ≥ 3 Gini points above the logistic scorecard) and SHAP-based explanations can be demonstrated to satisfy the institution's model risk governance framework, escalation to a hybrid or ensemble approach may be considered — but this requires explicit sign-off from risk management and legal counsel, not a purely technical decision.

---

## Data Sources & Features

The dataset used in this project contains applicant-level records including:

- **Basic applicant information:** Registration channel, registration length, number of bank cards, credit-to-debit ratio.
- **RFMS behavioral features:** For each of 10 spending behavior categories (Debit, Consumption, Consumption Loan, Transfer, Phone Bill, Utility Bill, Gaming, State-owned Bank, Medium Bank, VIP Card), we construct four variables: Recency (R), Frequency (F), Monetary average (M), and Standard deviation (S). This yields 40 derived behavioral features (Huang et al., 2018).
- **Company credit score:** A pre-existing general-purpose credit score developed by the originating institution.

Full data dictionary and preprocessing steps are documented in `notebooks/01_eda.ipynb`.

---

## Methodology

1. **Exploratory Data Analysis** — Distribution analysis, missing value assessment, default rate by segment.
2. **Proxy Variable Construction** — Definition and justification of the binary default proxy label.
3. **Weight of Evidence (WoE) Encoding** — Binning of continuous features, calculation of WoE and Information Value (IV) for variable selection.
4. **Logistic Regression Scorecard** — Model training, BIC-based variable selection, coefficient interpretation, scorecard scaling (400–800 point range).
5. **Ensemble Benchmark** — XGBoost / LightGBM trained on the same feature set; SHAP analysis for feature importance comparison.
6. **Model Evaluation** — ROC/AUC, Gini, Kolmogorov-Smirnov statistic, Population Stability Index, out-of-time validation.

---

## Results & Evaluation

*To be completed after model training. Results will include:*

- ROC curves for all model variants
- AUC and Gini comparison table
- Scorecard variable contributions (WoE coefficient chart)
- SHAP summary plot (gradient boosting benchmark)
- Population Stability Index over time

---

## References

- Huang, D., Zhou, J., & Wang, H. (2018). RFMS Method for Credit Scoring Based on Bank Card Transaction Data. *Statistica Sinica*, 28, 2903–2919. https://doi.org/10.5705/ss.202017.0043
- Hong Kong Monetary Authority (HKMA). (2021). *Alternative Credit Scoring of Micro-, Small and Medium-sized Enterprises*. https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf
- Basel Committee on Banking Supervision. (2001). *The Internal Ratings-Based Approach: Consultative Document*. Bank for International Settlements. https://www.bis.org/publ/bcbsca05.pdf
- Basel Committee on Banking Supervision. (2017). *Basel III: Finalising Post-Crisis Reforms*. Bank for International Settlements. https://www.bis.org/bcbs/publ/d424.pdf
- Corporate Finance Institute. *Credit Risk*. https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/
- Demir, C. (2026). Machine Learning for Credit Risk Scoring: From Traditional Statistics to Gradient Boosting. *Medium / Towards Data Science*. https://medium.com/@candemir13/machine-learning-for-credit-risk-scoring-from-traditional-statistics-to-gradient-boosting-95f056a1cc36
- World Bank Group. (2020). *Credit Scoring Approaches Guidelines*. https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf
- Miller, T. (2017). Explanation in Artificial Intelligence: Insights from the Social Sciences. *arXiv preprint* arXiv:1706.07269.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Model-Agnostic Interpretability of Machine Learning. *arXiv preprint* arXiv:1606.05386.

---


