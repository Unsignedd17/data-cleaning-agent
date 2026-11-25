import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"

data = pd.read_csv("titanic.csv")

summary = (
    "Column types:\n" + data.dtypes.to_string() + "\n\n"
    "Empty boxes in each column:\n" + data.isna().sum().to_string() + "\n\n"
    "First 5 rows:\n" + data.head().to_string()
)

question = (
    "You are a data cleaning assistant. Look at this table summary. "
    "For every column that has empty boxes, choose ONE action from this list only: "
    "fill_median, fill_mode, fill_unknown, drop_column. "
    "Reply with only a JSON object, with the column name as key and the action as value. "
    'Example: {"Age": "fill_median"}\n\n' + summary
)

reply = client.models.generate_content(model=MODEL, contents=question)
text = reply.text
plan = json.loads(text[text.find("{"): text.rfind("}") + 1])
print("AI plan:", plan)

for column, action in plan.items():
    if column not in data.columns:
        continue
    if action == "fill_median":
        data[column] = data[column].fillna(data[column].median())
    elif action == "fill_mode":
        data[column] = data[column].fillna(data[column].mode()[0])
    elif action == "fill_unknown":
        data[column] = data[column].fillna("Unknown")
    elif action == "drop_column":
        data = data.drop(columns=[column])
    else:
        print("Skipping unknown action:", action)

print()
print("Empty boxes left:")
print(data.isna().sum())

data.to_csv("agent_clean_titanic.csv", index=False)