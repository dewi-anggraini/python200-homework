# --- Mini-Project — World Happiness Agent ---
import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from smolagents import tool
from scipy.stats import pearsonr
from pathlib import Path
from smolagents import CodeAgent, OpenAIServerModel, tool

# --- Pre-task: Load the Data ---
# directory from which it is launched.
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_PATH = REPO_ROOT / "assignments_01" / "outputs" / "merged_happiness.csv"
YEARLY_DATA_DIR = REPO_ROOT / "assignments" / "resources" / "happiness_project"
OUTPUT_DIR = BASE_DIR / "outputs"
REGIONAL_PLOT_PATH = OUTPUT_DIR / "happiness_by_region.png"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared global DataFrame
df = None

print("DATA_PATH:", DATA_PATH)
print("Exists:", os.path.exists(DATA_PATH))

test_df = pd.read_csv(DATA_PATH)
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())

# --- Task 1: Define Your Tools ---

# Tool 1: load_happiness_data
# -------------------------------------------

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged World Happiness CSV from DATA_PATH. If the merged
    file does not exist, loads and merges all yearly CSV files from
    assignments/resources/happiness_project/.

    Returns:
        A dictionary containing the dataset shape and column names.
    """

    global df

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

    else:
        csv_files = sorted(
            glob.glob(str(YEARLY_DATA_DIR / "*.csv"))
        )
        #csv_files = sorted(
        #    glob.glob("assignments/resources/happiness_project/*.csv")
        #)
        #csv_files = sorted(
        #    glob.glob("assignments/resources/happiness_project/*.csv")
        #)

        if not csv_files:
            return {
                "error": "No CSV files found in assignments/resources/happiness_project/."
            }

        dfs = []

        for file_path in csv_files:
            yearly_df = pd.read_csv(file_path)

            # Standardize column names across yearly files
            if "Ladder score" in yearly_df.columns:
                yearly_df.rename(
                    columns={"Ladder score": "Happiness score"},
                    inplace=True
                )

            dfs.append(yearly_df)

        df = pd.concat(dfs, ignore_index=True)
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
    }    

# Tool 2: summarize_column
# -------------------------------------------

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary containing descriptive statistics for the requested
        column, or an error dictionary if the data is not loaded or the
        column does not exist.
    """
    if df is None:
        return {"error": "Data is not loaded."}
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    try:
        return df[column].describe().to_dict()
    except Exception as e:
        return {"error": str(e)}

# Tool 3: compute_correlation
# -------------------------------------------

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary containing col1, col2, pearson_r, and p_value,
        or an error dictionary if the data is not loaded or a column
        does not exist.
    """
    if df is None:
        return {"error": "Data is not loaded."}
    if col1 not in df.columns:
        return {"error": f"Column '{col1}' does not exist."}
    if col2 not in df.columns:
        return {"error": f"Column '{col2}' does not exist."}

    try:
        data = df[[col1, col2]].dropna()
        r, p_value = pearsonr(data[col1], data[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p_value, 4)
        }
    except Exception as e:
        return {"error": str(e)}

# Tool 4: get_top_n_countries
# -------------------------------------------

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column used to rank the countries.
        year: The year to filter the dataset by.
        n: The number of countries to return.

    Returns:
        A list of dictionaries containing the country name and value for
        the requested column, or an error dictionary if the data or
        requested inputs are invalid.
    """
    if df is None:
        return {"error": "Data is not loaded."}

    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    try:
        year_data = df[df["year"] == year]
        
        top_n = (
            year_data
            .sort_values(column, ascending=False)
            .head(n)
        )

        return [
            {
                "country": row["Country"],
                column: row[column]
            }
            for _, row in top_n.iterrows()
        ]

    except Exception as e:
        return {"error": str(e)}  


# --- Task 2: Build the Agent ----

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
) 

# --- Task 3: Run Guided Queries ---

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the Happiness score column.",
    "What is the correlation between GDP per capita and Happiness score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot Happiness score over the years as a line chart, with one line per Regional indicator. Save the plot to outputs/happiness_by_region.png."
]

if __name__ == "__main__":

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    plot_path = "outputs/happiness_by_region.png"

    if os.path.exists(plot_path):
        print(f"Plot saved successfully: {plot_path}")
    else:
        print(f"Plot was not found: {plot_path}")

# --- Task 4: Your Own Questions ---

 # My query 1
    my_query_1 = "Show me the top 3 countries with the lowest social support in 2019."  
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: This triggered tool use. The agent used the summarize_column tool.

    # My query 2
    my_query_2 = """
    Create a histogram of the Happiness score using the actual dataset.
    Use pandas to read assignments_01/outputs/merged_happiness.csv directly,
    then use matplotlib to create the histogram.
    Do not use mock or simulated data.
    Save the plot to outputs/happiness_histogram.png.
    """

    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
   # Comment: This triggered code generation. The agent wrote pandas and matplotlib
   # code to read the actual dataset and create the histogram.


# --- Task 5: Reflection ---

# 1. In Query 3, how did the agent communicate whether the correlation was
# statistically significant? Did it use the p-value correctly? What
# threshold did it apply?

# The agent reported a Pearson correlation = 0.6218 and a p-value of 0.0
# after rounding to four decimal places. It used the p-value to determine
# whether the correlation was statistically significant. Since the p-value
# was below the 0.05 significance threshold, the agent correctly concluded
# that the correlation between GDP per capita and Happiness score was
# statistically significant.


# 2. Did any of the agent's responses surprise you — either by being more
# capable than you expected, or less? Describe one specific example.

# I was surprised that the agent could write pandas and matplotlib code by
# itself to create charts. For example, when I asked it to create a histogram
# of the Happiness score, it wrote Python code to read the actual dataset and
# create the plot. This showed me that the agent can do more than just use
# the tools I created and can write its own code when the available tools
# are not enough.


# 3. What one additional tool would make this agent meaningfully more useful?

# I would add a tool to compare happiness scores for countries or regions
# across different years. The tool could take a country or region and a range
# of years and return the happiness scores over time. This would make it
# easier for the agent to answer questions about how happiness changed over
# time.

