from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from retriever import build_retrieve_tool


def build_graph(vector_store, index, llm):
    """Builds a state graph for handling question-answering tasks using LangGraph."""

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Handles the initial query or response generation."""
        llm_with_tools = llm.bind_tools([build_retrieve_tool(vector_store, index)])
        response = llm_with_tools.invoke(state["messages"])

        return {"messages": [response]}

    # Step 2: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generates a response based on retrieved content."""
         
         # Extract recent tool messages (retrieved documents)
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]  # Reverse to maintain order

        # Format the retrieved content into a system message
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know."
            "If the answer is not contained in the pieces of context, say that you don't know."
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            f"{docs_content}"
        )
        
        # Prepare the conversation messages for the LLM
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Generate the response
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Build the graph using LangGraph
    graph_builder = StateGraph(MessagesState)

    # Add nodes to the graph
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(ToolNode([build_retrieve_tool(vector_store, index)]))
    graph_builder.add_node(generate)

    # Define the entry point and edges
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Compile the graph with memory checkpointing
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph
    
# A function to query the graph
def query_graph(graph, config, input_message, history):
    """Queries the graph with a user input and updates the conversation history."""

    final_output = None


    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        final_output = step

    if final_output:
        history.append((input_message, final_output["messages"][-1].content))
    return "", history