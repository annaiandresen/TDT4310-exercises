import pandas as pd
import os.path

PATH: str = "reddit.l2/reddit.l2.clean.500K/reddit.{}.txt.tok.clean.shf.500K.nometa.tc.noent.fw.url.lc"
COUNTRIES: list = ["Finland", "France", "Norway", "Russia"]
DATASET_PATH: str = "data/dataset_small.pkl"


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
            self.save_to_file()

    def build(self) -> pd.DataFrame:
        """
        Reads all files in path.
        Combines all data into a single dataframe
        """
        df = pd.DataFrame(columns=["text", "label", "l1"])
        label = 0
        for country in self.countries:
            # Creates a dataframe with each country
            path = self.path.format(country)
            country_df = pd.read_csv(path, engine='python', encoding="utf-8", sep='\t', names=['text'])
            country_df['label'] = label
            country_df['l1'] = self.country_to_language(country)
            if self.small:
                # reduce size of dataset
                country_df = country_df.iloc[:10000, :]
                print(country_df)

            frames = [df, country_df]
            df = pd.concat(frames)
            label = label+1
        return df

    def save_to_file(self) -> None:
        try:
            self.df.to_pickle(self.ds_path)
            print(self.df)
            print("Dataframe saved to " + self.ds_path)
        except OSError:
            print("Something went wrong when saving dataframe")

    def load_dataset(self) -> pd.DataFrame:
        print("Loading dataframe from " + self.ds_path)
        return pd.read_pickle(self.ds_path)

    def shuffle(self):
        return self.df.sample(frac=1).reset_index(drop=True)

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