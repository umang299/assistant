import requests
from uuid import uuid4
import streamlit as st


endpoint = 'http://127.0.0.1:8001/'

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_session():
    st.session_state.messages = []
    for key, value in st.session_state.items():
        if not isinstance(value, list):
            st.session_state[key] = False


with st.sidebar:
    resp = requests.get(url=f'{endpoint}/repositories', timeout=300)    
    if resp.status_code == 200:
        repos = resp.json()
        repo = st.selectbox(
                    label='Select Repository',
                    index=None,
                    options=repos)

    with st.container():
        resp = requests.get(url=f'{endpoint}/threads', timeout=300)

        if resp.status_code == 200:
            threads = resp.json()

            def update_session_state(active_thread_id, threads):
                # Reset all buttons' states to False
                for thread_id in threads:
                    st.session_state[f'{thread_id}_click'] = False
                    st.session_state['messages'] = []
                # Set the clicked button's state to True
                st.session_state[f'{active_thread_id}_click'] = True
                conv_history_resp = requests.get(url=f'{endpoint}/history', json={'thread_id' : active_thread_id}, timeout=300)
                if conv_history_resp.status_code == 200:
                    st.session_state.messages = conv_history_resp.json()
            
            for i, id_ in enumerate(threads):
                # Initialize session state for each thread button
                if f'{id_}_click' not in st.session_state:
                    st.session_state[f'{id_}_click'] = False
                
                 # Create a button and attach the on_click handler
                st.button(
                    label=f'{id_}',
                    use_container_width=True,
                    on_click=update_session_state,
                    args=(id_, threads)  # Pass the thread ID to the function
                )
        
        st.button(label='New conversation', 
                  use_container_width=True,
                  on_click=reset_session)
        


thread_status = next((key for key, value in st.session_state.items() if value is True), None)
if thread_status is not None:
    thread_id = thread_status.split('_')[0]
else:
    thread_id = str(uuid4())


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    data = {
        "thread_id" : thread_id,
        "message" : prompt
    }
    response = requests.post(url=f'{endpoint}/response', json=data, timeout=300)
    if response.status_code == 200:

        # response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.json()['response'])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.json()['response']})