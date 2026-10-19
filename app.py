import os
import uuid
from anthropic import Anthropic
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

MODEL_NAME = "qwen3.8-flash"

openai =  Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url=os.environ["ANTHROPIC_BASE_URL"],
    default_headers={
        "x-opencode-session": str(uuid.uuid4()),
        "User-Agent": "real-madrid-chatbot/1.0",
    },
)

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]


def chat(message, history):
    messages = history + [{"role": "user", "content": message}]
    response = openai.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        system=TWIN_SYSTEM_PROMPT,
        messages=messages
    )
    while response.stop_reason == "tool_use":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
