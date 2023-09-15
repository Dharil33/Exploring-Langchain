import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title('Langchain Explorer')
input_text = st.text_input("Search a topic")

#Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

#Memory
person_memory = ConversationBufferMemory(input_key='name',memory_key='name_history')
dob_memory = ConversationBufferMemory(input_key='dob',memory_key='dob_history')


#OpenAI LLM Model
llm = OpenAI(temperature=0.8)

chain = LLMChain(llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain1 = LLMChain(llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

parent_chain = SequentialChain(chains=[chain,chain1],input_variables=['name'],
                               output_variables=['person','dob'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))
    
    with st.expander('Person name'):
        st.info(person_memory.buffer)
        
    with st.expander('Dob name'):
        st.info(dob_memory.buffer)
    
 