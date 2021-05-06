#Librairy to use streamlit

import streamlit as st
from streamlit_player import st_player

# Librairy to draw the graph
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
#%matplotlib inline



# Function to define the relation possible
def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1", [pattern]) 

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)



# Function to define the differente entities in a function
def get_entities(sent):
  ## chunk 1
  ent1 = " "
  ent2 = ""

  prv_tok_dep = " "    # dependency tag of previous token in the sentence
  prv_tok_text = " "   # previous token in the sentence

  prefix = " "
  modifier = " "

  #############################################################
  "compound", "prep", "conj", "mod"
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "det":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "obj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "mod":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          prefix = prv_tok_text + " "+ tok.text
      
      if tok.dep_ == "det":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "det":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "obj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "mod":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          prefix = prv_tok_text + " "+ tok.text


      if tok.dep_ == "obj":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "det":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "obj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "mod":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          prefix = prv_tok_text + " "+ tok.text


      if tok.dep_ == "prep":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "det":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "obj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "mod":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          prefix = prv_tok_text + " "+ tok.text



      if tok.dep_ == "conj":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "det":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "obj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "mod":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          prefix = prv_tok_text + " "+ tok.text



      
      if tok.dep_ == "mod":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "det":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "obj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "mod":
          prefix = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          prefix = prv_tok_text + " "+ tok.text



      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "det":
          modifier = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "dobj":
          modifier = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "prep":
          modifier = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "conj":
          modifier = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "pobj":
          modifier = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "cc":
          modifier = prv_tok_text + " "+ tok.text
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      
      
      if tok.dep_.endswith("prep") == True:
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "det":
          modifier = prv_tok_text 
        if prv_tok_dep == "dobj":
          modifier = prv_tok_text 
        #if prv_tok_dep == "prep":
          #modifier = prv_tok_text
        #if prv_tok_dep == "conj":
        #  modifier = prv_tok_text
        if prv_tok_dep == "pobj":
          modifier = prv_tok_text
        if prv_tok_dep == "cc":
          modifier = prv_tok_text
        if prv_tok_dep == "compound":
          modifier = prv_tok_text


      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = " "
        modifier = " "
        prv_tok_dep = " "
        prv_tok_text = " "      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]











# Main to draw the graph

description=[]
description.append("starteq I swim in the sea endseq")
description.append("starteq a girl work in my room endseq")
description.append("starteq I walk in the street endseq")
description.append("starteq a man play in my room endseq")
description.append("starteq I sing in the sea endseq")
description.append("starteq a boy sing in the street endseq")


descriptionclean=[]
descriptionclean.append(description[len(description)-5][9:(len(description[len(description)-5])-7)])
descriptionclean.append(description[len(description)-4][9:(len(description[len(description)-4])-7)])
descriptionclean.append(description[len(description)-3][9:(len(description[len(description)-3])-7)])
descriptionclean.append(description[len(description)-2][9:(len(description[len(description)-2])-7)])
descriptionclean.append(description[len(description)-1][9:(len(description[len(description)-1])-7)])

holderimage = st.beta_columns(1)


entity_pairs = []

for i in range(len(descriptionclean)):
     entity_pairs.append(get_entities(descriptionclean[i]))


relations = [get_relation(descriptionclean[i]) for i in range(len(descriptionclean))]

# extract subject
source = [i[0] for i in entity_pairs]
     
# extract object
target = [i[1] for i in entity_pairs]
     
#table of relation in the sentences
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
print(kg_df)

    
#Ploting of the graph
#fig = plt.figure(figsize=(12,12))
plt.figure(figsize=(12,12))

    # create a directed-graph from a dataframe
G = nx.from_pandas_edgelist(kg_df, "source", "target", 
                              edge_attr=True, create_using=nx.MultiDiGraph())

pos = nx.spring_layout(G)
nx.draw(G, pos, edge_color='black', width=1, linewidths=3,
            node_size=300, node_color='skyblue', alpha=1,node_shape="s",
            labels={node: node for node in G.nodes()})
edges = nx.get_edge_attributes(G, 'edge')
edge_labels = {i[0:2]:'{}'.format(i[2]['edge']) for i in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels,label_pos=0.5, font_size=12)
plt.savefig('langmodel22.png')
#st.write("Knowledge Graph")
#st.image("langmodel22.png")#, width=None)
#holderimage.header("Knowledge Graph")
#holderimage.image('langmodel22.png')
#shape=fig.get_size_inches()
#plt.show()
col1, col2 = st.beta_columns(2)
col1.header("Original")
col1.image("langmodel22.png", use_column_width=True)



"""st.title("Let's create a table!")
for i in range(1, 10):
    cols = st.beta_columns(4)
    cols[0].write(f'{i}')
    cols[1].write(f'{i * i}')
    cols[2].write(f'{i * i * i}')
    cols[3].write('x' * i)"""







