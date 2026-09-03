# AI-Powered Agricultural Commodity Market Prediction

An AI-driven agricultural market prediction and decision-support system that forecasts commodity prices, evaluates market opportunities, estimates travel distance and transportation costs, and calculates potential profit or loss to help farmers choose a suitable selling market.

---

## 📌 Overview

Agricultural commodity prices vary across markets and over time. For farmers, choosing a market solely based on price may not always result in the highest return because transportation distance and associated costs also affect the final profit.

This project provides a data-driven approach to this problem by combining **commodity price forecasting, market analysis, location processing, route estimation, transportation cost calculation, and profit/loss analysis**.

The system uses trained **commodity-specific LSTM models** to forecast prices and evaluates different markets based on the expected financial outcome.

---

## 🎯 Problem Statement

Farmers often need to answer questions such as:

- What price can I expect for my commodity?
- Which market offers a better selling opportunity?
- How far is each market from my location?
- How much will transportation cost?
- After transportation expenses, which market can provide better returns?

Traditional market selection based only on the current price does not account for transportation expenses or predicted future prices.

This project attempts to provide a more comprehensive market decision by considering these factors together.

---

## 💡 Proposed Solution

The system follows this general process:

```text
Farmer Information
       │
       ├── Commodity
       ├── Quantity
       └── Location
       │
       ▼
Historical Commodity Data
       │
       ▼
LSTM Price Forecasting
       │
       ▼
Predicted Commodity Price
       │
       ▼
Market Evaluation
       │
       ├── Market Price
       ├── Distance
       ├── Travel Time
       └── Transportation Cost
       │
       ▼
Revenue Calculation
       │
       ▼
Profit / Loss Estimation
       │
       ▼
Market Comparison
       │
       ▼
Selling Recommendation
```

## 🚀 Installation & Usage

### 1. Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/Vijaysinh-Bhosale/Crop2Market_lstm.git
```
Navigate to the project directory:
```bash
cd Crop2Market_lstm
```
### 2. Create a Virtual Environment

Create a Python virtual environment:
```bash
python -m venv venv
```
### 3. Activate the Virtual Environment
Windows
```bash
venv\Scripts\activate
```
macOS / Linux
```bash
source venv/bin/activate
```

After activation, your terminal should show (venv) before the current directory.

### 4. Install Dependencies

Install all required Python packages:
```
pip install -r requirements.txt
```
### 5. Run the Application

Start the Streamlit application:
```
streamlit run app.py
```
The application will open in your default browser.

If it does not open automatically, Streamlit will display a local URL similar to:
```
http://localhost:8501
```
Open that URL in your browser.

## 👨‍💻 Author

### **Vijaysinh Yuvarj Bhosale**

Machine Learning | Python | TensorFlow | LSTM | Streamlit | Data Analysis

### License
This project is developed for educational and demonstration purposes.
