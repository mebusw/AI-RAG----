import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Step 0: 加载环境变量
load_dotenv()
DIFY_URL = os.getenv("DIFY_URL", "http://118.195.145.124:8091/v1/chat-messages")
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

def buildUI():
    ## 布局
    st.title("IELTS雅思写作AI考官")
    st.sidebar.title("Chat History")

    app = st.session_state
    if 'messages' not in app:
        app['messages'] = [{"role": "assistant", "content": "你好！我是雅思写作AI考官，请提出你的写作题目或问题。"}] # Changed initial message
    if 'history' not in app:
        app['history'] = []
    if 'conversation_id' not in app: # Added for Dify
        app['conversation_id'] = ""
    if 'streaming_in_progress' not in app:
        app['streaming_in_progress'] = False
    if 'stop_streaming_flag' not in app:
        app['stop_streaming_flag'] = False

    ## 保持消息在聊天中
    for msg in app["messages"]:
        avatar = "🧑" if msg["role"] == "user" else "🤖" # Simplified avatar logic
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    ## 聊天
    if user_query := st.chat_input("请输入你的问题或写作任务..."): # Changed variable name for clarity
        ### 用户写入
        app["messages"].append({"role": "user", "content": user_query})
        st.chat_message("user", avatar="🧑",).write(user_query)
        
        current_full_response = "" # Accumulator for the current response
        

        # ### AI 使用聊天流式响应
        with st.chat_message("assistant", avatar="🤖"):
            app['streaming_in_progress'] = True
            if st.button("Stop Streaming", key="stop_button", disabled=not app['streaming_in_progress']):
                app['stop_streaming_flag'] = True
                ai.stop_streaming() # Stop the AI streaming
                app['streaming_in_progress'] = False # Indicate streaming is no longer in progress
                st.balloons("Streaming stopped by user.")

            response_placeholder = st.empty() # Create a placeholder
            for chunk in ai.respond(app["messages"], use_knowledge=True, conversation_id=app.get("conversation_id", "")):
                # print(f"Raw chunk from Dify: {chunk}") # For debugging the raw stream
                if chunk is not None:
                    # Dify might send keep-alive pings or other non-JSON lines
                    if not chunk.strip() or chunk.strip() == "[DONE]": # Handle empty lines or Dify's [DONE] signal
                        if chunk.strip() == "[DONE]":
                            print("Stream finished with [DONE]")
                        continue
                    if chunk.startswith("event: ping"):
                        continue

                    try:
                        # Remove prefix if present
                        json_string = chunk[len("data: "):] if chunk.startswith("data: ") else chunk
                        
                        
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
            
            response_placeholder.markdown(current_full_response, unsafe_allow_html=True) # Display final response without cursor

        ### 显示历史记录
        app["messages"].append({"role": "assistant", "content": current_full_response})
        app['history'].append("🧑: " + user_query)
        app['history'].append("🤖: " + current_full_response)
        # Consider limiting history display or using a more robust method if it gets very long
        st.sidebar.markdown("<br/>".join(app['history']) + "<br/><br/>", unsafe_allow_html=True)



class AI:
    def __init__(self):
        self.headers = {
            'Authorization': f'Bearer {DIFY_API_KEY}',
            'Content-Type': 'application/json',
        }
        self.stop_flag_key = False

    def stop_streaming(self):
        self.stop_flag_key = True

    def respond(self, lst_messages, use_knowledge=False, conversation_id=""):
        payload = {
            "inputs": {}, # Add any specific inputs Dify expects for your app
            "query": lst_messages[-1]["content"], # The actual user query
            "response_mode": "streaming",
            "user": "streamlit-user-dify-123", # A unique ID for the user
            "conversation_id": conversation_id, # Pass the current conversation ID
            "files": [], # Ensure this is an empty list if no files, not [{}]
        }

        try:
            response = requests.post(DIFY_URL, headers=self.headers, json=payload, stream=True) # Use json=payload
            response.raise_for_status() # Check for HTTP errors

            for line in response.iter_lines():
                # Check the stop flag in session_state
                if self.stop_flag_key:
                    break # Exit the loop if stop button was pressed
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
