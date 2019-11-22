import spacy
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt   

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_md")
texto = open('TheOutsider.txt',encoding='utf-8')
text = texto.read()
sentimentData= pd.read_csv('senticnet5.txt',sep="\t",index_col=0)
#coleccion={'INTENSITY':[]}
coleccion={}

# =============================================================================
# print(sentimentData)
# print(sentimentData.at["abandon","INTENSITY"])
# =============================================================================
# Process whole documents
# =============================================================================
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")
# =============================================================================
doc = nlp(text)
bookData=pd.DataFrame(coleccion)
pos=["SYM","PUNCT","X","SPACE"]

for token in doc:
    if (token.pos_ not in pos) & (token.text in sentimentData.index.values):
        aux={'INTENSITY':sentimentData.at[token.text,"INTENSITY"]}
        bookData = bookData.append(aux, ignore_index=True)
#print(bookData)
# =============================================================================
# plot function
plt.figure(figsize=(50,5))
sb.set(style='darkgrid')
line_plot=sb.scatterplot(x = bookData.index.values, y = "INTENSITY", data=bookData)
plt.savefig('bookScatterplot')
# =============================================================================

# =============================================================================
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
#     lexeme = doc.vocab[token.text]
#     print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_, lexeme.suffix_, lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)
# =============================================================================
# =============================================================================
# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
# 
# # Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)
# =============================================================================
