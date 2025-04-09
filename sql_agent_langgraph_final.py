#!/usr/bin/env python
# coding: utf-8

# In[5]:


import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from typing import Annotated, Literal
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase



# In[6]:


from langchain_community.utilities import SQLDatabase
import os

# Database credentials
db_user = "root"
db_password = ""
db_host = "127.0.0.1"  # Match Laravel's setup
db_name = "automobiz"

# Use pymysql instead of mysqlconnector
db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Connect to the database
db = SQLDatabase.from_uri(db_uri)
print(db)


# In[ ]:


db.dialect


# In[ ]:


db.get_usable_table_names()


# In[ ]:


from langchain_groq import ChatGroq
llm=ChatGroq(model="llama3-70b-8192",
             api_key="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" 
             )


# In[ ]:


llm.invoke("hello how are you?")


# In[ ]:


from langchain_community.agent_toolkits import SQLDatabaseToolkit


# In[ ]:


toolkit=SQLDatabaseToolkit(db=db, llm=llm)


# In[ ]:


tools=toolkit.get_tools()


# In[ ]:


tools


# In[ ]:


for tool in tools:
    print(tool.name)


# In[ ]:


list_tables_tool = next((tool for tool in tools if tool.name == "sql_db_list_tables"), None)


# In[ ]:


list_tables_tool


# In[ ]:


list_tables_tool.invoke("")


# In[ ]:


get_schema_tool = next((tool for tool in tools if tool.name == "sql_db_schema"), None)


# In[ ]:


get_schema_tool


# In[ ]:


print(get_schema_tool.invoke("vehicles"))


# In[ ]:


llm_to_get_schema=llm.bind_tools([get_schema_tool])


# In[ ]:


from langchain_core.tools import tool
@tool
def query_to_database(query:str)->str:
    """
    Execute a SQL query against the database and return the result.
    If the query is invalid or returns no result, an error message will be returned.
    In case of an error, the user is advised to rewrite the query and try again.
    """
    result=db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


# In[ ]:


query_to_database.invoke("SELECT * FROM vehicles;")


# In[ ]:


llm_with_tools=llm.bind_tools([query_to_database])


# In[ ]:


llm_with_tools.invoke("SELECT * FROM vehicles;")


# In[ ]:


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# In[ ]:


def handle_tool_error(state:State):
    error = state.get("error") 
    tool_calls = state["messages"][-1].tool_calls
    return { "messages": [ ToolMessage(content=f"Error: {repr(error)}\n please fix your mistakes.",tool_call_id=tc["id"],) for tc in tool_calls ] }

def create_node_from_tool_with_fallback(tools:list)-> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


# In[ ]:


list_tables=create_node_from_tool_with_fallback([list_tables_tool])
get_schema=create_node_from_tool_with_fallback([get_schema_tool])
query_database=create_node_from_tool_with_fallback([query_to_database])


# In[ ]:


from langchain_core.prompts import ChatPromptTemplate

query_check_system = """You are a SQL expert. Carefully review the SQL query for common mistakes, including:

Issues with NULL handling (e.g., NOT IN with NULLs)
Improper use of UNION instead of UNION ALL
Incorrect use of BETWEEN for exclusive ranges
Data type mismatches or incorrect casting
Quoting identifiers improperly
Incorrect number of arguments in functions
Errors in JOIN conditions

If you find any mistakes, rewrite the query to fix them. If it's correct, reproduce it as is."""

query_check_prompt = ChatPromptTemplate.from_messages([("system", query_check_system), ("placeholder", "{messages}")])

check_generated_query = query_check_prompt | llm_with_tools


# In[ ]:


check_generated_query.invoke({"messages": [("user", "SELECT * FROM vehicles LIMIT 5;")]})


# In[ ]:


check_generated_query.invoke({"messages": [("user", "SELECT +++ FROM vehicles LIMITs 5;")]})


# In[ ]:


class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")

llm_with_final_answer=llm.bind_tools([SubmitFinalAnswer])


# In[ ]:


# Add a node for a model to generate a query based on the question and schema
query_gen_system_prompt = """You are a SQL expert with a strong attention to detail.Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

1. DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

2. Output the SQL query that answers the input question without a tool call.

3. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.

4. You can order the results by a relevant column to return the most interesting examples in the database.

5. Never query for all the columns from a specific table, only ask for the relevant columns given the question.

6. If you get an error while executing a query, rewrite the query and try again.

7. If you get an empty result set, you should try to rewrite the query to get a non-empty result set.

8. NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

9. If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

10. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Do not return any sql query except answer. """

query_gen_prompt = ChatPromptTemplate.from_messages([("system", query_gen_system_prompt), ("placeholder", "{messages}")])

query_generator = query_gen_prompt | llm_with_final_answer


# In[ ]:


query_generator.invoke({"messages": [("can you fetch the data from vehicles table?")]})


# In[ ]:


def first_tool_call(state:State)->dict[str,list[AIMessage]]:
    print(f"state from first_tool_call: {state}")
    return{"messages": [AIMessage(content="",tool_calls=[{"name":"sql_db_list_tables","args":{},"id":"tool_abcd123"}])]}


# In[ ]:


def check_the_given_query(state:State):
    print(f"state from check the given query: {state}")
    return {"messages": [check_generated_query.invoke({"messages": [state["messages"][-1]]})]}


# In[ ]:


def generation_query(state:State):
    message = query_generator.invoke(state)
    print(f"state from generation_query: {state}")

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}


# In[ ]:


def should_continue(state:State):
    print(f"state from should_continue: {state}")
    messages = state["messages"]
    last_message = messages[-1]
    print("last message")
    print(last_message)
    if getattr(last_message, "tool_calls", None):
        print("sooo")
        return END
    elif last_message.content.startswith("Error:"):
        print("dooo")
        return "query_gen"
    else:
        print("oooo")
        return "correct_query"


# In[ ]:


def llm_get_schema(state:State):
    print("state from llm_get_schema", state)
    response = llm_to_get_schema.invoke(state["messages"])
    return {"messages": [response]}


# In[ ]:


workflow = StateGraph(State)
workflow.add_node("first_tool_call",first_tool_call)
workflow.add_node("list_tables_tool", list_tables)
workflow.add_node("get_schema_tool", get_schema)
workflow.add_node("model_get_schema", llm_get_schema)
workflow.add_node("query_gen", generation_query)
workflow.add_node("correct_query", check_the_given_query)
workflow.add_node("execute_query", query_database)


# In[ ]:


workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges("query_gen",should_continue,
                            {END:END,
                            "correct_query":"correct_query"})
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")


# In[ ]:


app=workflow.compile()


# In[ ]:


from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


# In[ ]:


query={"messages": [("user", "how many red cars are there in the vehicles table ?")]}


# In[ ]:


response=app.invoke(query)


# In[ ]:


response["messages"][-1].tool_calls[0]["args"]["final_answer"]


# In[ ]:


app.stream()

