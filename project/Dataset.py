import pandas as pd
import os.path
import re

PATH: str = "data/reddit.l2.raw/reddit.{}.txt.tok.clean"
COUNTRIES: list = ["Finland", "France", "Norway", "Russia"]
DATASET_PATH: str = "data/dataset.pkl"


class Dataset:
    def __init__(self, small: bool = True, path: str = PATH, countries: list = COUNTRIES, ds_path: str = DATASET_PATH):
        self.ds_path = ds_path
        self.small = small
        if os.path.exists(self.ds_path):
            self.df = self.load_dataset()
        else:
            self.path = path
            self.countries = countries
            self.df = self.build()
            self.clean()
            self.save_to_file()

    def build(self, nrows=40000) -> pd.DataFrame:
        """
        Reads all files in path.
        Combines all data into a single dataframe
        """
        df = pd.DataFrame(columns=["text", "label", "l1"])
        label = 0
        for country in self.countries:
            # Creates a dataframe with each country
            path = self.path.format(country)
            country_df = pd.read_csv(path, engine='python', nrows=nrows, encoding="utf-8", sep='\t', names=['text'])
            country_df['label'] = label
            country_df['l1'] = self.country_to_language(country)

            frames = [df, country_df]
            df = pd.concat(frames)
            label = label+1
        return df

    def save_to_file(self) -> None:
        try:
            self.df.to_pickle(self.ds_path)
            self.df.to_csv("data/dataset_small.csv")
            print(self.df)
            print("Dataframe saved to " + self.ds_path)
        except OSError:
            print("Something went wrong when saving dataframe")

    def load_dataset(self, from_csv=True) -> pd.DataFrame:
        print("Loading dataframe from " + self.ds_path)
        if from_csv:
            df = pd.read_csv(self.ds_path, usecols=['text', 'label', 'l1'])
            # Drop empty columns
            df = df.dropna()
            print(df.head(5))
            return df
        return pd.read_pickle(self.ds_path)

    def shuffle(self):
        return self.df.sample(frac=1).reset_index(drop=True)

    def clean(self):
        self.df.text = self.df.text.apply(lambda txt: self.clean_text(txt))

    @staticmethod
    def country_to_language(country: str) -> str:
        if country == "Albania" or country == "Russia":
            return country + "n"
        elif country == "China":
            return "Chinese"
        elif country == "Denmark":
            return "Danish"
        elif country == "Finland":
            return "Finnish"
        elif country == "Norway":
            return "Norwegian"
        elif country == "France":
            return "French"
        elif country == "Portugal":
            return "Portuguese"
        elif country == "Spain":
            return "Spanish"
        else:
            return country

    @staticmethod
    def clean_text(text: str):
        # Remove author name and subreddit
        text = text.strip()
        text_list = re.split("\[(.*?)\]", text)
        text_list = [txt for txt in text_list if txt and txt != " "]

        # Remove leading '>' for replies
        cleaned = text_list[2].strip()
        cleaned = cleaned.replace(">", "") if cleaned[0] == ">" else cleaned
        return cleaned
