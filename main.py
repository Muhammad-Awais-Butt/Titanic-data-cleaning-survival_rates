import pandas as pd
from pathlib import Path

def load_data():
    # ---------making directories and importing data----------------
    cur_dir = Path.cwd()
    data_dir = cur_dir / "data" / "raw"
    data_dir.mkdir(exist_ok=True)
    data = data_dir / "Titanic-Dataset.csv"

    df = pd.read_csv(data)
    print("Loading completed") 
    print(f"Successfully loaded {len(df)} rows") 
    print(f"Successfully loaded {len(df.columns)} columns")
    return df
    # ---------------------------------------------------------------
    

def fill_age(df):
    df["Age"] = df["Age"].fillna(df.groupby(["Pclass","Sex"])["Age"].transform("median"))
    return df

def fam_size(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df

def survival_rates(df):

    print("-"*25)
    print("Survival rates by class and gender in percentage")
    print((df.groupby(["Pclass","Sex"])["Survived"].mean() * 100).round(1))
    
    print("-"*25)
    print("Survival rates by gender in percentage")
    print((df.groupby("Sex")["Survived"].mean() * 100).round(1))

    print("-"*25)
    print("Survival rates by class in percentage")
    print((df.groupby("Pclass")["Survived"].mean() * 100).round(1))

def main():
    df = load_data()
    df = fill_age(df)
    df = fam_size(df)
    survival_rates(df)

if __name__ == "__main__":
    main()
