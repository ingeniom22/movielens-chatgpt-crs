import json
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv
import tensorflow as tf
from libreco.algorithms import DeepFM
from libreco.data import DataInfo
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
MODEL_PATH = "model"

with open("movie_id_mappings.json", "r") as json_file:
    movie_id_mappings = json.load(json_file)


tf.compat.v1.reset_default_graph()
data_info = DataInfo.load(MODEL_PATH, model_name="deepfm_model")
model = DeepFM.load(
    path=MODEL_PATH, model_name="deepfm_model", data_info=data_info, manual=True
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

class RecSysInput(BaseModel):
    uid: int = Field(description="User id")
    k: int = Field(description="number of recommended")


def recommend_top_k(uid: int, k: int):
    """Retrieve top k recommended movies for a User"""
    prediction = model.recommend_user(user=uid, n_rec=k, user_feats={"gender": "M", "occupation": 13, "age": 56}) # TODO: ambil user feature berdasarkan uid 
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
]

# Prompt Constructor
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah seorang pakar film yang sedang berbicara dengan user uid={current_user_uid}",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


known_user_uid = 896 # Jika uid ada pada training data
agent_executor.invoke(
    {
        "input": "Berikan saya 5 rekomendasi film!", # input yang diberikan user
        "current_user_uid": known_user_uid, # diambil dari Session
    },
)


unknown_user_uid = -123 # Jika uid belum ada pada training data, model bisa menggunakan user feature (gender, occupation, age)
agent_executor.invoke(
    {
        "input": "Berikan saya 5 rekomendasi film!", # input yang diberikan user
        "current_user_uid": unknown_user_uid, # diambil dari Session
    },
)