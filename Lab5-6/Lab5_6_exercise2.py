"""Exercise 2 - Link that entity!
Continue using the data from Exercise 1. With HTTP-calls (“requests” library) in Python, build functionality
to fetch information from entities using the WikiData knowledge base. This can be simplified by using
the “qwikidata” library found here: https://github.com/kensho-technologies/qwikidata. This task
is fairly open, but requires you to fetch either relationships (e.g. father, mother) or other info such as
aliases from entities in the text.
"""

import spacy
from Lab5_6_exercise1 import separate_into_chapters, Ner
import requests
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api


def lookup_entity(entity):
    url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json".format(
        entity.replace(" ", "_"))
    res = requests.get(url)
    return res.json() if res.status_code == 200 else {}


def get_entity_id(res):
    for key in res["search"]:
        if "character" in key["description"]:
            return key["id"]


class ExtendedNer:
    def __init__(self, text, entity=None):
        self.ner = Ner(text)
        self.entity = list(entity) if entity else list()
        self.lookups = {}
        if entity:
            self.lookup_entity(entity)

    def add_entity(self, ent):
        if ent not in self.entity:
            self.entity.append(ent)

    def add_entity_to_lookups(self, ent_key, *args):
        for val in args:
            self.ner.set_key(self.lookups, ent_key, val)

    @staticmethod
    def lookup_entity(entity):
        url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json".format(
            entity.replace(" ", "_"))
        res = requests.get(url)
        return res.json() if res.status_code == 200 else {}

    @staticmethod
    def get_entity_id(res):
        for key in res["search"]:
            if "character" in key["description"]:
                return key["id"]




if __name__ == '__main__':
    # nlp = spacy.load("en_core_web_sm")
    chapters = separate_into_chapters('chamber_of_secrets.txt')

    dic = lookup_entity("Harry Potter")
    entity_id = get_entity_id(dic)
    wiki = WikidataItem(get_entity_dict_from_api(entity_id))
    description = wiki.get_description()
