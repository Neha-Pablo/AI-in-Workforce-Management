![Project Banner](banner.png)
# Project 1: AI Forecasting for Staffing Demand

## Live Demo  
[Click here to launch the AI Forecasting Dashboard](https://ai-in-workforce-management-g2zfxmrgjkebwuvgbzhdrd.streamlit.app/)

_(Note: The link opens a live Streamlit dashboard built in Google Colab using Cloudflare. It may take ~15 seconds to load.)_

Goal: Build an AI based forecasting model to predict contact volume and convert forecasts into staffing requirements for workforce planning teams.

## Business problem
In large operations such as Amazon or Uber, accurate staffing forecasts drive SLA delivery, overtime costs, and agent morale. Under staffing leads to delays and missed targets. Over staffing increases idle time and cost. The challenge is to balance these forces by predicting demand precisely and converting it into optimal FTE requirements.

## Approach
1. Data simulation. A 120 day synthetic dataset with 30 minute intervals that includes weekday seasonality, intraday peaks, and a small growth trend.
2. Exploratory analysis. Daily and intraday visualizations to understand patterns.
3. Baseline models.
   - Last week same interval
   - Six interval moving average
4. Machine learning model. A Random Forest regressor trained on lag and calendar features to predict next interval volume.
5. Staffing model. Convert forecasted volume into FTE requirements using AHT, occupancy, and shrinkage.

## Results and insights
- Baseline forecasts captured general seasonality. The Random Forest improved precision for high variance intervals in testing.
- The notebook produces an automated staffing plan that translates forecasted volume into interval level FTE using operational assumptions.
- The workflow can be extended to multi site forecasting with historical call and attendance data by adding site and queue dimensions.

## Next steps
1. Add time series cross validation and hyperparameter tuning.
2. Add features for events such as outages, product launches, or promotions.
3. Wrap results in a Streamlit dashboard for scenario testing.
4. Parameterize AHT, shrinkage, and occupancy for what if analysis by site and queue.

## How to run
1. Open notebooks/01_staffing_forecast_baseline.ipynb and run all cells.
2. Adjust AHT, occupancy, and shrinkage in the staffing function to match your scenario.
3. The notebook exports docs/staffing_plan_example.csv as an example output.

## Folder contents
- datasets/staffing_forecast.csv
- notebooks/01_staffing_forecast_baseline.ipynb
- docs/staffing_plan_example.csv
