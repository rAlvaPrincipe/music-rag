import spacy


class NER:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
    
    
    # extracts only PERSON, WORK_OF_ART, ORG entities
    def get_entities(self, question):
        doc = self.nlp(question)
        
        entities = []
        for ent in doc.ents:
            print((ent.text, ent.label_))
            if ent.label_ == "ORG" or ent.label_ == "PERSON" or ent.label_ == "WORK_OF_ART":
                entities.append(ent.text)
        
        print("Entities considered: " + str(entities))
        return entities
    