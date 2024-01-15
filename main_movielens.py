# Import library yang diperlukan
import json
import tensorflow as tf
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
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

# Load konfigurasi dari file .env
load_dotenv()
MODEL_PATH = "model"

# Load mapping untuk ID film dari file JSON
with open("movie_id_mappings.json", "r") as json_file:
    movie_id_mappings = json.load(json_file)

# Reset graph TensorFlow yang ada
tf.compat.v1.reset_default_graph()

# Load informasi data dan model sistem rekomendasi
data_info = DataInfo.load(MODEL_PATH, model_name="deepfm_model")
model = DeepFM.load(
    path=MODEL_PATH, model_name="deepfm_model", data_info=data_info, manual=True
)


# Definisikan model input untuk sistem rekomendasi
class RecSysInput(BaseModel):
    uid: int = Field(description="User id")
    k: int = Field(description="Number of movies to be recommended")


# Fungsi untuk merekomendasikan film
def recommend_top_k(uid: int, k: int):
    """Retrieve top k recommended movies for a User"""
    prediction = model.recommend_user(
        user=uid, n_rec=k, user_feats={"gender": "M", "occupation": 13, "age": 56}
    )  # TODO: ambil user feature berdasarkan uid
    movie_ids = prediction[uid]
    movie_names = [movie_id_mappings[str(mid)] for mid in movie_ids]

    info = f"Rekomendasi film untuk user adalah {', '.join(movie_names)}"
    return info


# Konversi fungsi rekomendasi menjadi tool terstruktur
recsys = StructuredTool.from_function(
    func=recommend_top_k,
    name="RecSys",
    description="Retrieve top k recommended movies for a User",
    args_schema=RecSysInput,
    return_direct=False,
)

# List tools yang digunakan
tools = [
    recsys,
]

# Key untuk menyimpan riwayat obrolan dalam memory
MEMORY_KEY = "chat_history"

# Template prompt untuk percakapan
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah seorang pakar {role} yang sedang berbicara dengan user uid={current_user_uid}.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Inisialisasi model bahasa ChatGPT
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Bind tools ke model bahasa
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Konfigurasi agen percakapan
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
        "current_user_uid": lambda x: x["current_user_uid"],
        "role": lambda x: x["role"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# Inisialisasi executor untuk agen
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# User ID yang sudah diketahui (dapat diubah sesuai kebutuhan)
known_user_uid = 896

# Inisialisasi riwayat obrolan
chat_history = []

# Input peran dari pengguna
role = input("Masukkan Role: ")

# Loop utama percakapan
while True:
    # Input dari pengguna
    user_input = input("User: ")

    # Eksekusi agen untuk mendapatkan jawaban
    result = agent_executor.invoke(
        {
            "input": user_input,
            "current_user_uid": known_user_uid,
            "chat_history": chat_history,
            "role": role,
        }
    )

    # Tampilkan jawaban dari agen
    print(f"Agent: {result['output']}")

    # Update riwayat obrolan
    chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=result["output"]),
        ]
    )
