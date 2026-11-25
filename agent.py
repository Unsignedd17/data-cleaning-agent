import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"


def ask_ai(question):
    # If the AI is busy, wait and try again, up to 4 times
    for attempt in range(1, 5):
        try:
            return client.models.generate_content(model=MODEL, contents=question).text
        except Exception as error:
            print("Attempt", attempt, "failed:", str(error)[:100])
            time.sleep(20)
    raise SystemExit("The AI did not answer. Please try again later.")


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

text = ask_ai(question)
plan = json.loads(text[text.find("{"): text.rfind("}") + 1])
print("AI plan:", plan)

report = []

for column, action in plan.items():
    if column not in data.columns:
        continue

    is_number = pd.api.types.is_numeric_dtype(data[column])
    empty_percent = round(data[column].isna().mean() * 100)

    if action == "fill_median":
        if is_number:
            data[column] = data[column].fillna(data[column].median())
            report.append(column + ": filled with the median, because " + str(empty_percent) + "% was empty.")
        else:
            report.append(column + ": REFUSED fill_median, because this column is not numbers.")
    elif action == "fill_mode":
        data[column] = data[column].fillna(data[column].mode()[0])
        report.append(column + ": filled with the most common value.")
    elif action == "fill_unknown":
        data[column] = data[column].fillna("Unknown")
        report.append(column + ": filled with the word Unknown.")
    elif action == "drop_column":
        if empty_percent > 50:
            data = data.drop(columns=[column])
            report.append(column + ": dropped, because " + str(empty_percent) + "% was empty.")
        else:
            report.append(column + ": REFUSED drop_column, because only " + str(empty_percent) + "% was empty.")
    else:
        report.append(column + ": skipped unknown action " + str(action))

print()
print("What the agent did:")
for line in report:
    print("-", line)

print()
print("Empty boxes left:", int(data.isna().sum().sum()))

data.to_csv("agent_clean_titanic.csv", index=False)

with open("report.txt", "w") as file:
    file.write("\n".join(report))