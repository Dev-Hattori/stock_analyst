from langchain.chat_models import init_chat_model
from langchain.agents.middleware import SummarizationMiddleware


import streamlit as st
import uuid
from itertools import chain

from agent_backend import financial_analyst
from tools import get_income_statement, get_balance_sheet, get_cash_flows, \
    get_key_metrics_info, get_company_summary, get_ticker_news_sentiment

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Stock Fundamentals_Analyzer"

# --- Session State Initialization ---
# Initialize the thread_id if it doesn't exist
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Store conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Agent Setup ---
if "app" not in st.session_state:
    model = init_chat_model(
        model="ministral-3:3b",
        model_provider="ollama",
        temperature=0.2
    )

    tools = [get_income_statement, get_balance_sheet, get_cash_flows,
             get_key_metrics_info, get_company_summary, get_ticker_news_sentiment]

    system_prompt = """You are a Senior Equity Research Analyst. Your goal is to answer user questions about stock fundamentals using a specific set of financial tools.

    ### CRITICAL INSTRUCTIONS
    1.  **Conciseness is Key:** Your default response style must be **concise, data-backed, and to-the-point**. Do not fluff. Only provide a "Detailed Analysis" if the user explicitly asks for it.
    2.  **Tool Economy:** Do NOT call every tool for every request. Use only the specific tools required to answer the user's exact question.
        * *Example:* If asked for "current sentiment," use ONLY `get_ticker_news_sentiment`. Do not fetch the balance sheet.
    3.  **Chain of Thought:** You must perform a "thinking" step before answering. In this step, decide which tools are strictly necessary and outline your analysis.
    4. **No Hallucinations:** If a tool fails or returns incomplete data, state "Data unavailable for [metric]" instead of making up numbers.
    5. **Data Cutoff:** When mentioning data, always specify the date of the latest data point (e.g., "As of FY 2023...").
    
    ### YOUR TOOLKIT
    * `get_company_summary`: For business model, sector, and industry context.
    * `get_income_statement`: For revenue, net income, margins, and profitability trends.
    * `get_balance_sheet`: For assets, liabilities, debt levels, and liquidity (current ratio).
    * `get_cash_flows`: For operating cash flow and Free Cash Flow (FCF) analysis.
    * `get_key_metrics_info`: For valuation ratios (P/E, EV/EBITDA) and efficiency (ROE).
    * `get_ticker_news_sentiment`: For current market mood, recent headlines, and qualitative risks.

    ### REASONING PROCESS (Internal)
    Before calling tools or answering, you must output a `<thinking>` block:
    1.  **Analyze Request:** What is the user specifically asking for?
    2.  **Select Tools:** Which tools are *absolutely* needed? (If none, answer from knowledge).
    3.  **Plan:** How will I synthesize this data into a concise answer?

    ### RESPONSE GUIDELINES
    * **Standard Mode (Default):** Give a direct answer + 2-3 bullet points of supporting data. Total length < 100 words.
    * **Detailed Mode (Only if requested):** Provide an Executive Summary, Financial Breakdown, Valuation, and Risk Assessment.
    * **Missing Data:** If a tool fails, state "Data unavailable for [metric]" instead of hallucinating numbers.

    ### FORMAT
    <thinking>
    [Your internal reasoning and tool selection logic here]
    </thinking>

    [Your final concise response here]"""

    summarization_model = init_chat_model(
        model="gemma3:1b",
        model_provider="ollama",
        temperature=0.2
    )

    middleware = [
        SummarizationMiddleware(
            model=summarization_model,
            trigger=("tokens", 1000000),
            keep=("messages", 10),
            system_prompt="You are a helpful assistant that summarizes conversation history to save tokens while retaining important information. Summarize previous messages concisely, focusing on key points relevant to ongoing discussion about stock fundamental analysis. Omit any redundant or less important details."
        )
    ]

    analyst_agent = financial_analyst(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware
    )

    st.session_state.app = analyst_agent


# --- Streamlit Framework ---
st.title('💰 Stock Fundamentals Analyst')
st.caption(f"Conversation ID: `{st.session_state.thread_id}`")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input(
    "Ask a question about a stock ticker (e.g., 'What is the P/E ratio for AAPL?')...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the agent
    agent_output = st.session_state.app.analyze_stream(
        user_input, thread_id=st.session_state.thread_id)
    first_token = ""
    with st.spinner("Analyzing fundamentals..."):
        try:
            first_token = next(agent_output)
        except StopIteration:
            st.warning("The agent did not return any response.")

    output_stream = chain([first_token], agent_output)

    # Display agent response and add to history
    final_response_content = ""
    with st.chat_message("assistant"):
        final_response_content = st.write_stream(output_stream)
    st.session_state.messages.append(
        {"role": "assistant", "content": final_response_content})
