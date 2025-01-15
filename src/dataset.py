def get_dataset(name):
    if name == "giocattolo":
        return dataset_giocattolo()
    
    
def dataset_giocattolo():
    domande_risposta = [
        {"q": "Did Pink Floyd influence Radiohead?", 
         "a": "Pink Floyd influenced Radiohead through their pioneering use of atmospheric soundscapes, concept albums, and experimental approaches to music production. Radiohead's albums like OK Computer and Kid A reflect Pink Floyd's influence in their blending of rock with electronic elements, thematic depth, and innovative studio techniques."},
        
        {"q": "Who is the bassist of Radiohead?", 
         "a": "The bassist of Radiohead is Colin Greenwood."},
        
  #      {"q": "Where are the Sex Pistols from?", 
  #       "a": "The Sex Pistols are from London, England."},
        
  #      {"q": "Which band is known for its rivalry with Blur?", 
  #       "a": "Blur and Oasis had a famous rivalry in the 1990s Britpop scene. Blur's eclectic sound clashed with Oasis's anthemic rock, peaking in 1995 when Oasis's *Morning Glory* outsold Blur's *The Great Escape*."}
    ]

    return domande_risposta