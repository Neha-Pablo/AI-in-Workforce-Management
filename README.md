![Project Banner](banner.png)
# Project 1: AI Forecasting for Staffing Demand

## Live Demo  
[Click here to launch the AI Forecasting Dashboard](https://ai-in-workforce-management-g2zfxmrgjkebwuvgbzhdrd.streamlit.app/)

_(Note: The link opens a live Streamlit dashboard built in Google Colab using Cloudflare. It may take ~15 seconds to load.)_


### Project Overview

Workforce planning often relies on yesterday’s data to solve today’s problems. This project replaces static Excel-based forecasts with a learning model that adapts to seasonality and trend. It predicts contact volumes at 30-minute intervals and converts them into staffing requirements using operational inputs such as Average Handle Time (AHT), occupancy, and shrinkage. The goal is to make workforce management more adaptive, accurate, and efficient.

This project demonstrates how AI can automate forecasting, optimize staffing, and help analysts focus on strategy instead of manual recalculation.

---

### Key Features

| Feature                     | What it Does                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| Upload / Sample dataset     | Allows drag-and-drop uploads or demo data for instant testing                                |
| EDA section                 | Displays daily trend, intraday pattern, and weekday pattern                                  |
| AI vs Baseline forecast     | Compares a Random Forest model to a simple last-week baseline                                |
| Dynamic staffing calculator | Converts forecasts into FTE requirements using user inputs for AHT, occupancy, and shrinkage |
| Download button             | Exports a clean CSV with timestamp, forecast, and FTE requirement                            |
| Help expander               | Provides quick usage instructions directly in the app                                        |

---

### Business Problem and Solution

| Traditional Workforce Management      | AI Forecasting Approach                               |
| ------------------------------------- | ----------------------------------------------------- |
| Static averages in Excel              | Adaptive machine learning model that learns from data |
| Manual recalculation                  | Automatic retraining and faster updates               |
| Frequent over or under-staffing       | Dynamic forecasting adjusts to real patterns          |
| Disconnected forecasting and staffing | Integrated forecast-to-FTE conversion                 |
| Limited what-if analysis              | Interactive testing of AHT, occupancy, and shrinkage  |

---

### Dashboard Walkthrough

**Sidebar Controls**

* **Average Handle Time (AHT):** Average seconds per contact. Higher AHT increases required FTE.
* **Occupancy:** Proportion of agent time productively used. Lower occupancy means higher staffing needs.
* **Shrinkage:** Percentage of paid time not available for work (meetings, breaks, absenteeism).

**Model Performance (MAE Values):**

* *Baseline MAE:* Error using simple last-week logic.
* *AI Model MAE:* Error using Random Forest regression.
* *Improvement %:* Percentage improvement of AI over baseline.

**Charts Section**

1. **Daily Trend:** Shows total volume per day to identify long-term growth or dips.
2. **Intraday Pattern:** Displays average calls by hour to reveal peak load hours.
3. **Weekday Pattern:** Shows differences in workload by day of the week.
4. **Forecast vs Actual:** Compares AI predictions with actual data. Close overlap indicates good accuracy.
5. **FTE Requirement:** Converts predicted volume into required agents for each 30-minute interval.

**Download Output**
Exports a CSV with columns for timestamp, forecast, and FTE requirement. This file can be used directly in scheduling systems such as Verint or Kronos.

---

### Execution Environment

This project runs as an interactive **Streamlit dashboard** that can be launched either in **Google Colab** using Cloudflare for live preview or locally in any Python 3 environment with the required libraries.

Dependencies include:

* Python 3
* Streamlit
* pandas
* numpy
* scikit-learn
* matplotlib

---

### Results and Insights

* The Random Forest model improved forecast precision for high-variance intervals.
* AI-generated forecasts adapt better to volume spikes compared to static averages.
* The exported staffing plan translates forecasts directly into actionable staffing data.
* This project demonstrates how AI can bridge the gap between forecasting accuracy and operational efficiency.

---

### Real-World Relevance

This simulation reflects how large-scale operations such as Amazon, Uber, and Fidelity plan staffing across multiple sites. In such environments, even a small forecasting improvement reduces SLA breaches, overtime, and idle time significantly. By integrating forecasting with staffing logic, this model illustrates how AI can support decision-intelligent workforce planning.

---

### Author

**Neha Korati**
MBA, Operations and Supply Chain | University at Buffalo
Passionate about using AI and analytics to make operations smarter and more adaptive.
LinkedIn: [https://www.linkedin.com/in/neharikakorati/](https://www.linkedin.com/in/neharikakorati/)


