"""Exercise 2 - Link that entity!
Continue using the data from Exercise 1. With HTTP-calls (“requests” library) in Python, build functionality
to fetch information from entities using the WikiData knowledge base. This can be simplified by using
the “qwikidata” library found here: https://github.com/kensho-technologies/qwikidata. This task
is fairly open, but requires you to fetch either relationships (e.g. father, mother) or other info such as
aliases from entities in the text.
"""

from Lab5_6_exercise1 import separate_into_chapters, Ner
import requests
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
from collections import OrderedDict
import json
import os.path


class ExtendedNer:
    def __init__(self, text):
        self.ner = Ner(text)
        self.entities = self.ner.get_entities()
        if os.path.exists('entities.json'):
            self.lookups = self.load_file()
        else:
            print('Reading chapter 1 of Harry Potter and the Philosophers Stone, please wait...')
            self.lookups = {}
            self.create_dict_with_ids()

    def add_entity_to_lookups(self, ent_key):
        res = self.lookup_entity(ent_key)
        entity_id = self.get_entity_id(res)
        if entity_id:
            wiki = WikidataItem(get_entity_dict_from_api(entity_id))
            aliases = map(lambda x: str(x).lower(), wiki.get_aliases())
            aliases = list(OrderedDict.fromkeys(aliases))
            aliases.append(wiki.get_label().lower())
            self.ner.set_key(self.lookups, entity_id, aliases)
            return True
        else:
            return False

    def get_wiki_description(self, entity):
        for key, val in self.lookups.items():
            if entity in val:
                entity_id = key
                wiki = WikidataItem(get_entity_dict_from_api(entity_id))
                return wiki.get_description()
        return None

    def create_dict_with_ids(self):
        for entity in self.get_entities():
            res = self.lookup_entity(entity)
            entity_id = self.get_entity_id(res)
            if entity_id:
                self.add_entity_to_lookups(entity_id)

    @staticmethod
    def lookup_entity(entity):
        url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json".format(
            entity.replace(" ", "_"))
        res = requests.get(url)
        return res.json() if res.status_code == 200 else {}

    @staticmethod
    def get_entity_id(res):
        try:
            for key in res["search"]:
                if 'Harry Potter' in key["description"]:
                    return key["id"]
        except KeyError:
            return None

    def get_entities(self):
        str_entities = map(lambda x: str(x).lower(), self.entities)
        return tuple(OrderedDict.fromkeys(str_entities))

    def write_to_file(self, path="entities.json"):
        with open(path, 'w') as out:
            json.dump(self.lookups, out)
            out.close()
            print('File saved to', path)

    def load_file(self, path="entities.json"):
        with open(path) as file:
            data = json.load(file)
            print('Loading file...')
            file.close()
            return data


if __name__ == '__main__':
    chapter_1 = separate_into_chapters('chamber_of_secrets.txt')[0][24:]
    ner = ExtendedNer(chapter_1)
    print('Welcome to Harry Potter-Pedia 🧙\nType q to quit')
    while True:
        entity = input("Type a character from the Harry Potter series you want to look up \n").strip().lower()
        if entity == 'q':
            ner.write_to_file()
            break
        desc = ner.get_wiki_description(entity)
        if not desc:
            print("entity not in dictionary. adding entity...")
            added = ner.add_entity_to_lookups(entity)
            if added:
                print(entity, ner.get_wiki_description(entity), sep=" is a ")
            else:
                print('Cant find this character. Try a different character')
        else:
            print(entity, desc, sep=" is a ")

