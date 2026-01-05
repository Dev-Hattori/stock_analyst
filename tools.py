from langchain.tools import tool

import yfinance as yf
import requests
import os


@tool
def get_income_statement(ticker: str) -> str:
    """
    Useful for analyzing a company's profitability over time.
    Returns the Income Statement, including Revenue, Cost of Revenue, Gross Profit, Operating Expenses, and Net Income.
    Use this to answer questions about: Sales growth, margins, or earnings.
    """
    try:
        stock = yf.Ticker(ticker)

        income_statement = stock.financials

        finances = "Income Statement Table:\n" + income_statement.to_string()

        return finances
    except Exception as e:
        return f"Error fetching income statement for {ticker}: {str(e)}"


@tool
def get_balance_sheet(ticker: str):
    """
    Useful for analyzing a company's financial health at a specific point in time.
    Returns the Balance Sheet, including Assets (Cash, Inventory), Liabilities (Debt, Payables), and Shareholder Equity.
    Use this to answer questions about: Debt levels, cash position, or book value.
    """
    try:
        stock = yf.Ticker(ticker)

        balance_sheet = stock.balance_sheet

        finances = "Balance Sheet Table:\n" + balance_sheet.to_string()

        return finances
    except Exception as e:
        return f"Error fetching balance sheet for {ticker}: {str(e)}"


@tool
def get_cash_flows(ticker: str):
    """
    Useful for analyzing how cash enters and leaves the company.
    Returns the Cash Flow Statement, including Operating Cash Flow, Investing (Capex), and Financing activities.
    Use this to answer questions about: Free Cash Flow, burn rate, or capital expenditures.
    """
    try:
        stock = yf.Ticker(ticker)

        cash_flow = stock.cashflow

        finances = "Cash Flow Table:\n" + cash_flow.to_string()

        return finances
    except Exception as e:
        return f"Error fetching cash flows for {ticker}: {str(e)}"


@tool
def get_key_metrics_info(ticker: str):
    """
    Useful for getting a current snapshot of the stock's valuation and market data.
    Returns a dictionary containing: Market Cap, P/E Ratio, Dividend Yield, 52 Week High/Low, and Forward PE.
    Use this to answer questions about: 'Is the stock expensive?', current valuation, or dividend info.
    """
    try:
        stock = yf.Ticker(ticker)

        info = stock.info

        finances = "Key Informations:\n" + str(info)

        return finances
    except Exception as e:
        return f"Error fetching key metrics for {ticker}: {str(e)}"


@tool
def get_company_summary(ticker: str):
    """
    Useful for understanding what a company does.
    Returns the sector, industry, and a long business summary.
    Use this as the first step when the user asks about a new ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown Sector')
        summary = info.get('longBusinessSummary', 'No summary available.')
        # Truncate summary to save tokens if necessary
        return f"Sector: {sector}\nSummary: {summary[:500]}..."
    except Exception as e:
        return f"Error fetching summary for {ticker}: {str(e)}"


@tool()
def get_ticker_news_sentiment(ticker: str):
    """
    Fetches top 10 most relevant news along with sentiment data for a given stock ticker.
    """

    def get_relevance_score(x, ticker):
        for sentiment in x:
            if sentiment['ticker'] == ticker:
                return sentiment['relevance_score']
        return 0

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": os.getenv("alpha_vantage_api_key"),
        "limit": 50
    }
    response = requests.get(url, params=params)
    response = response.json()
    sorted_news = sorted(response['feed'], key=lambda x: get_relevance_score(
        x['ticker_sentiment'], ticker), reverse=True)

    news = []
    for item in sorted_news[:10]:
        ts = [i for i in item['ticker_sentiment'] if i['ticker'] == ticker][0]
        news.append(f"{item['title']}\nTime of publishing: {item['time_published']}\nAuthors: {",".join(item['authors'])}\nSummary: {item['summary']}\noverall sentiment: {item['overall_sentiment_score']}\noverall sentiment: {item['overall_sentiment_label']}\nticker sentiment: {ts['ticker_sentiment_score']}\nticker sentiment label: {ts['ticker_sentiment_label']}")

    return "\n\n".join(news)
