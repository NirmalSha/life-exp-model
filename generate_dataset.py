import pandas as pd
import numpy as np

np.random.seed(42)
ROWS = 10_000

def generate_data(n):
    data = []

    for _ in range(n):
        age = np.random.randint(18, 90)
        gender = np.random.choice(["male", "female"])
        bmi = round(np.random.normal(26, 4), 1)

        smoker = np.random.binomial(1, 0.3)
        alcohol = np.random.binomial(1, 0.4)

        diabetes = np.random.binomial(1, 0.2)
        heart = np.random.binomial(1, 0.15)
        cancer = np.random.binomial(1, 0.05)
        hyper = np.random.binomial(1, 0.25)
        asthma = np.random.binomial(1, 0.1)

        region = np.random.choice(["asia", "europe", "africa", "americas"])
        healthcare = np.random.choice(["poor", "average", "good"])

        # Base life expectancy
        life = 85

        # Age impact
        life -= age * 0.3

        # Lifestyle impact
        life -= smoker * 5
        life -= alcohol * 3
        life -= (bmi - 22) * 0.5

        # Disease impact
        life -= diabetes * 6
        life -= heart * 8
        life -= cancer * 12
        life -= hyper * 4
        life -= asthma * 2

        # Healthcare impact
        if healthcare == "good":
            life += 5
        elif healthcare == "poor":
            life -= 5

        life = max(30, round(life, 1))

        data.append([
            age, gender, bmi, smoker, alcohol,
            diabetes, heart, cancer, hyper, asthma,
            region, healthcare, life
        ])

    columns = [
        "age", "gender", "bmi", "smoker", "alcohol",
        "diabetes", "heart_disease", "cancer",
        "hypertension", "asthma",
        "region", "healthcare_access", "life_expectancy"
    ]

    return pd.DataFrame(data, columns=columns)


df = generate_data(ROWS)
df.to_csv("life_expectancy.csv", index=False)
print("Dataset generated: life_expectancy.csv")

