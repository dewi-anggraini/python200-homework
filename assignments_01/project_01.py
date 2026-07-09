
import pandas as pd
from prefect import task, flow, get_run_logger
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

# --- Task 1: Load Multiple Years ---
@task(
    retries=3,
    retry_delay_seconds=2
)
def load_happiness_data(file_paths):

    logger = get_run_logger()
    logger.info("Starting data loading...")

    dataframes = []

    for file_path in file_paths:

        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=","
        )

        # standarized 2024 column name
        if "Ladder score" in df.columns:
            df = df.rename(
                columns={"Ladder score": "Happiness score"}
        )

        year = file_path.split("_")[-1].replace(".csv", "")
        df["year"] = int(year)

        dataframes.append(df)

    merged_df = pd.concat(
        dataframes,
        ignore_index=True
    )
    # ensuring the outputs folder has been created
    os.makedirs("outputs", exist_ok=True)

    merged_df.to_csv(
        "outputs/merged_happiness.csv",
        index=False
    )

    return merged_df

@flow
def happiness_pipeline():

    file_paths = [
        f"resources/happiness_project/world_happiness_{year}.csv"
        # f"../python-200-v1-main/assignments/resources/happiness_project/world_happiness_{year}.csv"
        for year in range(2015, 2025)
    ]

    df = load_happiness_data(file_paths)
    descriptive_statistics(df)
    create_visualizations(df)
    hypothesis_testing(df)
    correlation_analysis(df)
    generate_summary(df)

    logger = get_run_logger()
    logger.info(f"\n{df.head()}")

# --- Task 2: Descriptive Statistics ---
@task
def descriptive_statistics(df):

    logger = get_run_logger()

    # Overall statistics
    mean_score = df["Happiness score"].mean()
    median_score = df["Happiness score"].median()
    std_score = df["Happiness score"].std()

    logger.info(f"Overall mean happiness: {mean_score}")
    logger.info(f"Overall median happiness: {median_score}")
    logger.info(f"Overall standard deviation: {std_score}")

    # Mean happiness by year
    happiness_by_year = (
        df.groupby("year")["Happiness score"]
        .mean()
    )

    logger.info("Mean happiness by year:")
    logger.info(f"\n{happiness_by_year}")

    # Mean happiness by region
    happiness_by_region = (
        df.groupby("Regional indicator")["Happiness score"]
        .mean()
        .sort_values(ascending=False)
    )

    logger.info("Mean happiness by region:")
    logger.info(f"\n{happiness_by_region}")

    return {
        "overall_mean": mean_score,
        "overall_median": median_score,
        "overall_std": std_score,
        "by_year": happiness_by_year.to_dict(),
        "by_region": happiness_by_region.to_dict()
    }

# --- Task 3: Visual Exploration ---
@task
def create_visualizations(df):

    logger = get_run_logger()

    # Histogram
    plt.figure(figsize=(8,5))

    plt.hist(
        df["Happiness score"],
        bins=20,
        edgecolor="black"
    )

    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")

    plt.savefig("outputs/happiness_histogram.png")

    plt.close()

    logger.info("Saved happiness_histogram.png")

    # Boxplot
    plt.figure(figsize=(10,6))

    sns.boxplot(
        data=df,
        x="year",
        y="Happiness score"
    )

    plt.title("Happiness Scores by Year")

    plt.savefig("outputs/happiness_by_year.png")

    plt.close()

    logger.info("Saved happiness_by_year.png")

    # Scatter Plot
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="GDP per capita",
        y="Happiness score"
    )

    plt.title("GDP vs Happiness")

    plt.savefig("outputs/gdp_vs_happiness.png")

    plt.close()

    logger.info("Saved gdp_vs_happiness.png")

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig("outputs/correlation_heatmap.png")

    plt.close()

    logger.info("Saved correlation_heatmap.png") # log a message

# --- Task 4: Hypothesis Testing ---
@task
def hypothesis_testing(df):

    logger = get_run_logger()

    # Create two groups:
    scores_2019 = df[df["year"] == 2019]["Happiness score"]

    scores_2020 = df[df["year"] == 2020]["Happiness score"]

    # Run the t-test
    t_stat, p_value = ttest_ind(
        scores_2019,
        scores_2020
    )

    # Log the means for each group
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info(f"2019 mean: {mean_2019:.3f}")
    logger.info(f"2020 mean: {mean_2020:.3f}")

    # Log the test result
    logger.info(f"T-statistic: {t_stat:.3f}")
    logger.info(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        logger.info(
            "There is statistically significant evidence that the average happiness score differed between 2019 and 2020."
        )
    else:
        logger.info(
            "There is no statistically significant evidence that the average happiness score differed between 2019 and 2020."
        )
    # By Region
    east_asia = df[
        df["Regional indicator"] == "East Asia"
    ]["Happiness score"]

    latin_america_and_caribbean = df[
        df["Regional indicator"] == "Latin America and Caribbean"
    ]["Happiness score"]

    t_stat, p_value = ttest_ind(
        east_asia,
        latin_america_and_caribbean,
        equal_var=False
    )

    logger.info("----- East Asia VS Latin America and Caribbean -----")
    logger.info(f"Mean happiness (East Asia): {east_asia.mean():.3f}")
    logger.info(f"Mean happiness (Latin America and Caribbean): {latin_america_and_caribbean.mean():.3f}")
    logger.info(f"T-statistic: {t_stat:.3f}")
    logger.info(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        logger.info(
            "There is statistically significant evidence that East Asia and Latin America and Caribbean have different average happiness scores."
        )
    else:
        logger.info(
            "There is no statistically significant evidence that East Asia and Latin America and Caribbean have different average happiness scores."
        )

# --- Task 5: Correlation and Multiple Comparisons ---
@task
def correlation_analysis(df):

    logger = get_run_logger()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    results = []

    for column in numeric_columns:

        if column == "Happiness score":
            continue

        clean_data = df[[column, "Happiness score"]].dropna()

        if len(clean_data) < 2:
            logger.info(f"Skipping {column}: not enough data for correlation.")
            continue

        correlation, p_value = pearsonr(
            clean_data[column],
            clean_data["Happiness score"]
        )

        logger.info(
            f"{column}: correlation={correlation:.3f}, "
            f"p-value={p_value:.4f}"
        )

        results.append({
            "variable": column,
            "correlation": correlation,
            "p_value": p_value
        })


    number_of_tests = len(results)

    adjusted_alpha = 0.05 / number_of_tests

    logger.info(
        f"Number of tests: {number_of_tests}"
    )

    logger.info(
        f"Bonferroni adjusted alpha: {adjusted_alpha:.5f}"
    )

    for result in results:

        original = result["p_value"] < 0.05

        corrected = (
            result["p_value"] < adjusted_alpha
        )

        logger.info(
            f"{result['variable']} | "
            f"original significant={original} | "
            f"Bonferroni significant={corrected}"
        )

# --- Task 6: Summary Report ---
@task
def generate_summary(df):

    logger = get_run_logger()

    # -----------------------------------
    # 1. Total countries and years
    # -----------------------------------

    total_countries = df["Country"].nunique()
    total_years = df["year"].nunique()

    logger.info(
        f"Dataset contains {total_countries} countries "
        f"across {total_years} years."
    )

    # -----------------------------------
    # 2. Top 3 and bottom 3 regions
    # -----------------------------------

    regional_means = (
        df.groupby("Regional indicator")["Happiness score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_3_regions = regional_means.head(3)
    bottom_3_regions = regional_means.tail(3)

    logger.info(
        f"Top 3 happiest regions:\n{top_3_regions}"
    )

    logger.info(
        f"Bottom 3 happiest regions:\n{bottom_3_regions}"
    )

    # -----------------------------------
    # 3. 2019 vs 2020 t-test summary
    # -----------------------------------

    scores_2019 = (
        df[df["year"] == 2019]["Happiness score"]
    )

    scores_2020 = (
        df[df["year"] == 2020]["Happiness score"]
    )

    t_stat, p_value = ttest_ind(
        scores_2019,
        scores_2020,
        equal_var=False
    )

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    if p_value < 0.05:

        logger.info(
            f"The average happiness score changed significantly "
            f"between 2019 ({mean_2019:.3f}) and 2020 "
            f"({mean_2020:.3f}). "
            f"This suggests the beginning of the pandemic was "
            f"associated with a change in global happiness levels."
        )

    else:

        logger.info(
            f"There was no statistically significant difference "
            f"between happiness scores in 2019 ({mean_2019:.3f}) "
            f"and 2020 ({mean_2020:.3f}). "
            f"The data does not provide strong evidence that "
            f"global happiness changed at the beginning of the pandemic."
        )

    logger.info(
        f"2019 vs 2020 t-statistic: {t_stat:.3f}"
    )

    logger.info(
        f"2019 vs 2020 p-value: {p_value:.4f}"
    )


    # -----------------------------------
    # 4. Strongest correlation after
    #    Bonferroni correction
    # -----------------------------------

    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
    )

    correlation_results = []

    for column in numeric_columns:

        # Skip happiness score itself
        if column == "Happiness score":
            continue

        correlation, p_value = pearsonr(
            df[column],
            df["Happiness score"]
        )

        correlation_results.append(
            {
                "variable": column,
                "correlation": correlation,
                "p_value": p_value
            }
        )

    number_of_tests = len(correlation_results)

    adjusted_alpha = 0.05 / number_of_tests

    significant_correlations = [
        result
        for result in correlation_results
        if result["p_value"] < adjusted_alpha
    ]

    if significant_correlations:

        strongest = max(
            significant_correlations,
            key=lambda x: abs(x["correlation"])
        )

        logger.info(
            f"The variable most strongly correlated with "
            f"happiness score after Bonferroni correction was "
            f"{strongest['variable']} "
            f"(correlation={strongest['correlation']:.3f})."
        )
    else:

        logger.info(
            "No variables remained significantly correlated "
            "with happiness score after Bonferroni correction."
        )
if __name__ == "__main__":
    happiness_pipeline()
