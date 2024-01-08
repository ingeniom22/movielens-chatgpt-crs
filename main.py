import json

import tensorflow as tf
from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
)
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import AIMessage, HumanMessage
from libreco.algorithms import DeepFM
from libreco.data import DataInfo
from pydantic.v1 import BaseModel, Field

load_dotenv()
MODEL_PATH = "model"

with open("movie_id_mappings.json", "r") as json_file:
    movie_id_mappings = json.load(json_file)

tf.compat.v1.reset_default_graph()
data_info = DataInfo.load(MODEL_PATH, model_name="deepfm_model")
model = DeepFM.load(
    path=MODEL_PATH, model_name="deepfm_model", data_info=data_info, manual=True
)


class RecSysInput(BaseModel):
    uid: int = Field(description="User id")
    k: int = Field(description="Number of movies to be recommended")


def recommend_top_k(uid: int, k: int):
    """Retrieve top k recommended movies for a User"""
    prediction = model.recommend_user(
        user=uid, n_rec=k, user_feats={"gender": "M", "occupation": 13, "age": 56}
    )  # TODO: ambil user feature berdasarkan uid
    movie_ids = prediction[uid]
    movie_names = [movie_id_mappings[str(mid)] for mid in movie_ids]

    info = f"Rekomendasi film untuk user adalah {', '.join(movie_names)}"
    return info


recsys = StructuredTool.from_function(
    func=recommend_top_k,
    name="RecSys",
    description="Retrieve top k recommended movies for a User",
    args_schema=RecSysInput,
    return_direct=False,
)


tools = [
    
    recsys,
    # human_input
]

# Prompt Constructor
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Anda adalah seorang pakar film yang sedang berbicara dengan user uid={current_user_uid}",
#         ),
#         ("user", "{input}"),
#         (MessagesPlaceholder(variable_name="agent_scratchpad")),

#     ]
# )

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah seorang pakar film yang sedang berbicara dengan user uid={current_user_uid}.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
        "current_user_uid": lambda x: x["current_user_uid"]
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

known_user_uid = 896  # Jika uid ada pada training data
chat_history = []

while True:
    user_input = input()
    result = agent_executor.invoke(
        {
            "input": user_input,
            "current_user_uid": known_user_uid,
            "chat_history": chat_history,
        }
    )
    
    chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=result["output"]),
        ]
    )

agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})

# agent_executor.invoke(
#     {
#         "input": "Berikan saya 5 rekomendasi film!", # input yang diberikan user
#         "current_user_uid": known_user_uid, # diambil dari Session
#     },
# )


# unknown_user_uid = -123 # Jika uid belum ada pada training data, model bisa menggunakan user feature (gender, occupation, age)
# agent_executor.invoke(
#     {
#         "input": "Berikan saya 5 rekomendasi film!", # input yang diberikan user
#         "current_user_uid": unknown_user_uid, # diambil dari Session
#     },
# )
