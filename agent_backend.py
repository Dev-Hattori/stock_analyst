from langchain.agents import create_agent
from langchain.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver


class financial_analyst:
    def __init__(self, model, tools, system_prompt, middleware=None):
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
