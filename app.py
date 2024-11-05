import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers



## Function

def getllmresponse(input_text: str, no_of_words: str, blog_style: str) -> str:
    """
    Description:
        Takes the input data from the user and generates the llm response
        from the saved model.
    Args:
        input_text: Defines the topic to be used
        no_of_words: Defines the length of the blog.
        blog_style: Defins the target audience.
    Returns:
        Returns the LLM response.
    """

    llm = CTransformers(model = "model/llama-2-7b-chat.ggmlv3.q8_0.bin", 
                        model_type= "llama",
                        config= {"max_new_tokens": 256,
                                 "temperature": 0.01})
    

    # tempelate = "Write a blog for {blog_style} profile for a topic {input_text} within {no_of_words} words."
    prompt = PromptTemplate(template="Write a blog post for {blog_style} about {input_text} within {no_of_words} words.", 
                            input_variables=["blog_style", "input_text", "no_of_words"])

    
    response = llm(prompt.format(blog_style = blog_style, 
                                 input_text = input_text, no_of_words = no_of_words))
    print(response)
    return response


def get_streamlit_interface():
    """
    Description:
        Generates a UI interface for the input text
    Returns:
        Returns the submit status, input text, blog style and length of the blog.
    """

    st.set_page_config(page_title="My LLM",
                    page_icon="\U0001F602",
                    layout='centered',
                    initial_sidebar_state="collapsed")

    st.header("My LLM \U0001F602")
    input_text = st.text_input("Enter thr Topic Name")

    ## Creating columns to add more 2 fields
    col1, col2 = st.columns([5,5])

    with col1:
        no_of_words = st.text_input("No. of Words")

    with col2:
        blog_style = st.selectbox("Writing the text for", ("Researchers", "Data Scientist", "Comman People"), index=0)

    submit = st.button("Generate")

    return submit, input_text, no_of_words, blog_style

    




if __name__ == "__main__":
    submit, input_text, no_of_words, blog_style = get_streamlit_interface()
    ## Final Response
    if submit:
        st.write(getllmresponse(input_text, no_of_words, blog_style))
    