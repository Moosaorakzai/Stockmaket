# 📈 Real-Time Stock Market Dashboard

A comprehensive, interactive stock market dashboard built with Streamlit that provides real-time analysis and visualization of stock market data.

## Features

- **Real-time Stock Data**: Pull the latest stock data from Yahoo Finance API
- **Interactive Charts**: Visualize historical stock prices, performance comparisons, and technical indicators
- **Multiple Analysis Views**: Four main tabs for different types of analysis:
  - Price Overview: Historical stock prices and recent data
  - Performance Comparison: Compare multiple stocks' percentage returns
  - Technical Analysis: Moving averages and candlestick charts
  - Predictive Trends: Simple linear regression-based price predictions

- **User Controls**: Customize your analysis with date range selectors, multiple stock selection, and technical indicator options

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd stock-market-dashboard
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Select the stocks you wish to analyze from the sidebar
2. Choose your desired date range
3. Explore the different tabs to view various analyses
4. Customize technical indicators and prediction parameters as needed

## Troubleshooting

### API Rate Limiting

The application uses Yahoo Finance API which may have rate limits. If you encounter issues:

- The app includes automatic retry logic with exponential backoff
- Reduce the number of stocks being analyzed simultaneously
- Avoid frequent refreshes of the page

### Data Loading Issues

If stock data fails to load:

- Check your internet connection
- Verify that the selected stock tickers are valid
- Try selecting a different date range

## Deployment

### Deploying to Streamlit Cloud

1. Create a GitHub repository for your project
2. Push your code to the GitHub repository
3. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
4. Connect your GitHub repository to Streamlit Cloud
5. Deploy the app by selecting the repository and the main file (app.py)

## Technologies Used

- **Streamlit**: For building the interactive web application
- **yfinance**: For fetching real-time stock data from Yahoo Finance
- **Pandas**: For data manipulation and analysis
- **Plotly**: For interactive data visualization
- **scikit-learn**: For predictive analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Data provided by Yahoo Finance
- Built with Streamlit 