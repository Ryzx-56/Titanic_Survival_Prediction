import pandas as pd 

def preprocess(path):

  df = pd.read_csv(path, index_col="PassengerId")
  
  # drop whole cabin column cause its mostly empty
  df = df.drop(columns=["Cabin"])
  # drop the ticket column cause its mixed with letters and numbers
  df = df.drop(columns=["Ticket"])

  # fill misiing values.
  #used the mode because its categorical 
  mode_value = df["Embarked"].mode()[0]
  df["Embarked"] = df["Embarked"].fillna(mode_value)

  df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))

  # New column called family size
  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

  #new column called is alone . checks if each person is alone
  df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

  # extracts the title from the name
  df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

  # Drop the name column cause its useless
  df = df.drop(columns=["Name"])

  # drop Sibsp and Parch because i have family size
  df = df.drop(columns=["SibSp", "Parch"])

  # groups rare titles together
  title_counts = df["Title"].value_counts()
  rare_titles = title_counts[title_counts < 10].index
  df["Title"] = df["Title"].replace(rare_titles, "Rare")

  # Start encoding for sex , Embarked , and Title
  df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title"], drop_first=True)

  return df



