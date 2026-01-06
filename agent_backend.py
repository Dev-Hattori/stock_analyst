from langchain.agents import create_agent
from langchain.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

default_system_prompt = """You are a Senior Equity Research Analyst. Your goal is to answer user questions about stock fundamentals using a specific set of financial tools.

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


class financial_analyst:
    def __init__(self, model, tools, system_prompt=default_system_prompt, middleware=None):
        """
        Initializes a Financial Analyst agent with the given language model.
        Args:
            model: The language model to be used by the agent.
            tools: The tools available to the agent.
            system_prompt: The system prompt for the agent.
        """
        self.llm = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = InMemorySaver()
        self.middleware = middleware
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware
        )

    def analyze(self, user_message, thread_id='default_thread'):
        """
        Analyzes the user's message using the Financial Analyst agent.
        Args:
            user_message: The message from the user to be analyzed.
            thread_id: The thread ID for maintaining conversation context.
        """
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": thread_id}}
        )
        return response['messages'][-1].content

    def analyze_stream(self, user_message, thread_id='default_thread'):
        """
        Analyzes the user's message using the Financial Analyst agent in a streaming manner.
        Args:
            user_message: The message from the user to be analyzed.
            thread_id: The thread ID for maintaining conversation context.
        """
        for token, metadata in self.agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages"
        ):
            if token.content:
                yield token.content

    def update_system_prompt(self, new_prompt):
        """
        Updates the system prompt dynamically.
        """
        self.system_prompt = new_prompt
        # Re-create the agent with the new prompt
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware
        )

    def get_history(self, thread_id='default_thread'):
        """
        Retrieves the conversation history for a specific thread.
        Useful for reloading chat context in a UI.
        """
        config = {"configurable": {"thread_id": thread_id}}
        current_state = self.agent.get_state(config)
        return current_state.values.get("messages", [])

    # def clear_history(self, thread_id='default_thread'):
    #     """
    #     Clears the conversation history for a specific thread.
    #     """

    def get_tool_metadata(self):
        """
        Returns a list of dictionaries containing tool names and descriptions.
        """
        return [{"name": t.name, "description": t.description} for t in self.tools]
