import streamlit as st
import requests
import json
# from langchain.callbacks.base import BaseCallbackHandler # Not used
# from typing import Generator # Not used
# from queue import Queue # Not used


def buildUI():
    ## 布局
    st.title("IELTS雅思写作AI考官")
    st.sidebar.title("Chat History")

    app = st.session_state
    if 'messages' not in app:
        app['messages'] = [{"role": "assistant", "content": "你好！我是雅思写作AI考官，请提出你的写作题目或问题。"}] # Changed initial message
    if 'history' not in app:
        app['history'] = []
    # 'full_response' will be built per response, so initializing it here might not be necessary
    # if 'full_response' not in app:
    #     app['full_response'] = ''
    if 'conversation_id' not in app: # Added for Dify
        app['conversation_id'] = ""


    ## 保持消息在聊天中
    for msg in app["messages"]:
        avatar = "🧑" if msg["role"] == "user" else "🤖" # Simplified avatar logic
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    ## 聊天
    if user_query := st.chat_input("请输入你的问题或写作任务..."): # Changed variable name for clarity
        ### 用户写入
        app["messages"].append({"role": "user", "content": user_query})
        st.chat_message("user", avatar="🧑").write(user_query)
        
        current_full_response = "" # Accumulator for the current response
        
        # ### AI 使用聊天流式响应
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty() # Create a placeholder
            
            for chunk in ai.respond(app["messages"], use_knowledge=True, conversation_id=app.get("conversation_id", "")):
                # print(f"Raw chunk from Dify: {chunk}") # For debugging the raw stream
                if chunk is not None:
                    PREFIX = "data: "
                    # Dify might send keep-alive pings or other non-JSON lines
                    if not chunk.strip() or chunk.strip() == "[DONE]": # Handle empty lines or Dify's [DONE] signal
                        if chunk.strip() == "[DONE]":
                            print("Stream finished with [DONE]")
                        continue

                    try:
                        # Remove prefix if present
                        json_string = chunk[len(PREFIX):] if chunk.startswith(PREFIX) else chunk
                        
                        # Attempt to parse JSON
                        chunk_json = json.loads(json_string)
                        # print(f"Parsed chunk_json: {chunk_json}") # For debugging parsed JSON

                        # Update conversation_id if present in the chunk
                        if "conversation_id" in chunk_json and chunk_json["conversation_id"]:
                            app["conversation_id"] = chunk_json["conversation_id"]
                            # print(f"Updated conversation_id: {app['conversation_id']}")

                        if "answer" in chunk_json and chunk_json["answer"] is not None:
                            current_full_response += str(chunk_json["answer"])
                            response_placeholder.markdown(current_full_response + "▌") # Update placeholder, add cursor
                        # Handle other event types if necessary, e.g., message_end might contain final conv_id
                        elif chunk_json.get("event") == "message_end":
                            if "conversation_id" in chunk_json and chunk_json["conversation_id"]:
                                app["conversation_id"] = chunk_json["conversation_id"]
                                # print(f"Final conversation_id from message_end: {app['conversation_id']}")
                            break # Often a good place to break if Dify sends a specific end event

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for chunk: '{chunk}'. Error: {e}")
                    except Exception as e:
                        print(f"An unexpected error processing chunk: '{chunk}'. Error: {e}")
            
            response_placeholder.markdown(current_full_response) # Display final response without cursor

        ### 显示历史记录
        app["messages"].append({"role": "assistant", "content": current_full_response})
        app['history'].append("🧑: " + user_query)
        app['history'].append("🤖: " + current_full_response)
        # Consider limiting history display or using a more robust method if it gets very long
        st.sidebar.markdown("<br/>".join(app['history']) + "<br/><br/>", unsafe_allow_html=True)


class AI:
    def __init__(self):
        # It's good practice to define constants like URL and API Key at a class or module level
        # or pass them during initialization if they can change.
        self.DIFY_URL = "http://118.195.145.124:8091/v1/chat-messages"
        self.DIFY_API_KEY = "app-OY2WgsPvHexb17EumGVW0JQi"
        self.headers = {
            'Authorization': f'Bearer {self.DIFY_API_KEY}',
            'Content-Type': 'application/json',
        }

    def respond(self, lst_messages, use_knowledge=False, conversation_id=""):
        # The prompt logic seems specific to a RAG setup, might not be directly used by Dify
        # Dify usually takes the raw query.
        # if use_knowledge:
        #     prompt = "Give the most accurate answer using your knowledge to user's query.\n'{query}':"
        # else:
        #     prompt = "Give the most accurate answer without external knowledge to user's query.\n'{query}':"

        payload = {
            "inputs": {}, # Add any specific inputs Dify expects for your app
            "query": lst_messages[-1]["content"], # The actual user query
            "response_mode": "streaming",
            "user": "streamlit-user-dify-123", # A unique ID for the user
            "conversation_id": conversation_id, # Pass the current conversation ID
            "files": [], # Ensure this is an empty list if no files, not [{}]
        }

        try:
            response = requests.post(self.DIFY_URL, headers=self.headers, json=payload, stream=True) # Use json=payload
            response.raise_for_status() # Check for HTTP errors

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    yield decoded_line
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"data: {json.dumps({'answer': f'Error: Could not connect to Dify. {e}', 'event': 'error'})}" # Yield an error message in Dify's format
        except Exception as e:
            print(f"An unexpected error occurred in respond: {e}")
            yield f"data: {json.dumps({'answer': f'Error: An unexpected error occurred. {e}', 'event': 'error'})}"


if __name__ == "__main__":
    ai = AI()
    buildUI()
