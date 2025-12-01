import gradio as gr
import requests

# Base URL of the local Llama.cpp server (assumes the server can load different models via the "model" field)
SERVER_URL = "http://localhost:8000/v1/chat/completions"

def _query_model(model_path: str, messages: list) -> str:
    """Send a chat request to the server for a specific model.

    Args:
        model_path: Path or identifier of the GGUF model to load.
        messages:   List of message dicts following OpenAI chat format.

    Returns:
        The assistant's reply as a string.
    """
    payload = {
        "model": model_path,
        "messages": messages,
    }
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def dual_chat(model_path_1: str, model_path_2: str, topic: str, turns: int = 5) -> str:
    """Generate a conversation between two models on a given topic.

    The function alternates messages: Model 1 → Model 2 → Model 1 … for *turns* exchanges.
    The initial user message is the *topic* supplied by the user.
    """
    # Initialise conversation log
    conversation = []
    # First user message – the topic
    user_msg = topic.strip()
    conversation.append(f"**User (topic):** {user_msg}")

    # Alternate between the two models
    for _ in range(turns):
        # Model 1 responds to the latest user message
        msgs = [
            {"role": "system", "content": "You are Model 1, a helpful assistant."},
            {"role": "user", "content": user_msg},
        ]
        reply1 = _query_model(model_path_1, msgs)
        conversation.append(f"**Model 1:** {reply1}")

        # Model 2 responds to Model 1's reply
        msgs = [
            {"role": "system", "content": "You are Model 2, a helpful assistant."},
            {"role": "user", "content": reply1},
        ]
        reply2 = _query_model(model_path_2, msgs)
        conversation.append(f"**Model 2:** {reply2}")

        # Next round: the next user message becomes Model 2's reply
        user_msg = reply2

    return "\n\n".join(conversation)

# Gradio UI – inputs for two model paths, a topic, and number of turns
iface = gr.Interface(
    fn=dual_chat,
    inputs=[
        gr.Textbox(label="Model 1 GGUF Path", placeholder="models/model1.gguf"),
        gr.Textbox(label="Model 2 GGUF Path", placeholder="models/model2.gguf"),
        gr.Textbox(label="Topic / Starting Prompt", lines=2, placeholder="Enter a discussion topic..."),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Turns", value=5),
    ],
    outputs=gr.Textbox(label="Conversation"),
    title="Dual‑Model GGUF Chatbot",
    description="Provide two GGUF model paths and a topic. The app will let the two models converse back‑and‑forth on the topic.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
