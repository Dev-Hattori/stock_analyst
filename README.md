# 📈 Stock Fundamentals Analyst Agent

An AI-powered financial research assistant capable of analyzing stock fundamentals, valuation metrics, and market sentiment. 

This project demonstrates a modular **Agentic Architecture**:
* **Backend (`agent_backend.py`):** A fully **model-agnostic** agent class built with LangChain & LangGraph. It can accept *any* compatible LLM (OpenAI, Anthropic, Llama, etc.).
* **Frontend (`app.py`):** A reference implementation using **Streamlit** configured to run locally with **Ollama** for cost-effective, private inference.

## 🚀 Features

* **Model Agnostic Core:** The `financial_analyst` class is decoupled from the specific LLM provider. While the demo uses local models, you can easily inject GPT-4, Claude, or Gemini into the backend.
* **Local Inference Demo:** The provided Streamlit frontend is pre-configured to run entirely locally using Ollama (`ministral-3:3b`), ensuring privacy and zero API costs for the LLM reasoning.
* **Token Optimization:** Integrated `SummarizationMiddleware` automatically condenses conversation history to manage context windows effectively, ensuring long-running conversations don't crash or become expensive.
* **Robust Toolkit:** A suite of 6+ specialized tools to fetch real-time financial data, including income statements, balance sheets, and news sentiment.
* **Agentic Workflow:** Utilizes a "Chain of Thought" reasoning process where the agent plans its analysis, selects the necessary tools, and synthesizes data into concise answers.

## 🛠️ Tech Stack

* **Orchestration:** [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Default Model Provider:** [Ollama](https://ollama.com/) (for the frontend demo)
* **Data Sources:**
    * [yfinance](https://pypi.org/project/yfinance/): Stock financials and historical data.
    * [Alpha Vantage](https://www.alphavantage.co/): News sentiment analysis.

## 📂 Project Structure

```text
├── app.py                  # Streamlit frontend (configured for Ollama by default)
├── agent_backend.py        # Agnostic agent logic, class definition, and middleware
├── tools.py                # Definitions of financial analysis tools
├── requirements.txt        # Project dependencies
├── .env                    # API keys and configuration
├── experimentation.ipynb   # R&D and testing playground
└── example.ipynb           # Examples of using the backend with different configurations

## ⚙️ Setup & Installation
### 1. Prerequisites
- Python 3.10+
- API Keys:
  - Alpha Vantage API Key (Required for news sentiment tool).
  - LangSmith API Key (Optional, recommended for tracing agent thoughts).
- Ollama (For App Demo):
  - Because app.py is configured for local inference, you must have Ollama installed to run the web interface as-is.
  - Note: If you plan to use OpenAI/Anthropic instead, you can skip installing Ollama and modify app.py.

### 2. Install Models (For Local Demo)
The frontend uses ministral-3:3b for the agent and gemma3:1b for summarization. Pull them via your terminal:
```Bash:
ollama pull ministral-3:3b
ollama pull gemma3:1b
```
### 3. Installation
```Bash:
git clone <your-repo-url>
cd stock-fundamentals-analyst
pip install -r requirements.txt
```
### 4. Configuration
Create a `.env` file in the root directory:
```Ini, TOML:
# .env
alpha_vantage_api_key="YOUR_ALPHA_VANTAGE_KEY"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_KEY"
LANGCHAIN_PROJECT="Stock Fundamentals_Analyzer"
```
## 🖥️ Usage

### Option A: Run the Streamlit Web App (Local LLM)
This will launch the web interface using the models pulled in step 2.
```Bash:
streamlit run app.py
```
- **URL:** `http://localhost:8501`
- **Capabilities:** Ask about profitability, P/E ratios, or compare companies (e.g., "Compare MSFT and GOOGL").
-
### Option B: Use the Backend with Other Models
You can import the `financial_analyst` class and use it with **OpenAI, Anthropic**, or any other LangChain chat model.
```Python:
from langchain_openai import ChatOpenAI
from agent_backend import financial_analyst
from tools import get_income_statement, get_key_metrics_info # ... import tools

# 1. Initialize ANY model (e.g., GPT-4o)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define Tools
tools = [get_income_statement, get_key_metrics_info]

# 3. Instantiate Agent (Backend is model-agnostic)
agent = financial_analyst(model=model, tools=tools, system_prompt="You are a financial analyst...")

# 4. Run Analysis
response = agent.analyze("What is the P/E of AAPL?")
print(response)
```

## 🧰 Available Tools

| Tool Name | Description | Source |
| :--- | :--- | :--- |
| `get_company_summary` | Fetches sector, industry, and a business summary. | `yfinance` |
| `get_income_statement` | Returns Revenue, Gross Profit, and Net Income trends. | `yfinance` |
| `get_balance_sheet` | Provides Assets, Liabilities, and Debt levels. | `yfinance` |
| `get_cash_flows` | Analyzes Operating Cash Flow and Free Cash Flow (FCF). | `yfinance` |
| `get_key_metrics_info` | Retrieves P/E Ratio, Market Cap, Dividend Yield, etc. | `yfinance` |
| `get_ticker_news_sentiment` | Fetches top 10 news articles and sentiment scores. | `Alpha Vantage` |

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
