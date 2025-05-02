import os
import sys
import requests
from openai import OpenAI
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import InMemoryHistory
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# Load .env file
load_dotenv()

# Initialize OpenRouter client (API key will be set dynamically)
client = None
api_key = os.getenv("OPENROUTER_API_KEY")  # Load from .env or system env

# Available models (initially static, updated dynamically with /free or /addmodel)
MODELS = {
    "deepseek-r1": "deepseek/deepseek-r1",
    "glm-4-32b": "thudm/glm-4-32b-0414",
    "nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "olympic-coder": "open-r1/olympiccoder-32b",
    "grok": "x-ai/grok-3-beta",
    "GPT": "openai/gpt-4o-search-preview",
    "claude": "anthropic/claude-3.7-sonnet",
    "eva": "eva-unit-01/eva-llama-3.33-70b"
}

# Current model (default to None until set)
current_model = "grok"  # Changed to a reasoning-capable model

# Markdown rendering
console = Console()
pretty_markdown = False  # Default: disabled

# Conversation and Reasoning modes
chat_mode = False  # Default: disabled
reasoning_mode = False  # Default: disabled
reasoning_effort = "medium"  # Default: medium (options: low, medium, high)
conversation_history = []  # Store conversation messages

def initialize_client() -> bool:
    """Initialize or update the OpenRouter client with the current API key."""
    global client
    if not api_key:
        print("Error: No API key set. Add OPENROUTER_API_KEY to .env or use /key <your-api-key>.")
        return False
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        return True
    except Exception as e:
        print(f"Error initializing client: {e}")
        return False

def clear_screen():
    os.system("clear")

def fetch_free_models() -> Optional[dict]:
    """Fetch free models from OpenRouter's /api/v1/models endpoint."""
    if not api_key:
        print("Error: No API key set for fetching models.")
        return None
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        free_models = {model["id"].split(":")[0]: model["id"] for model in models if ":free" in model["id"]}
        return free_models
    except Exception as e:
        print(f"Error fetching free models: {e}")
        return None

def add_non_free_model(model_id: str, set_as_current: bool = False) -> bool:
    """Fetch and validate a model from OpenRouter and add it to MODELS."""
    global MODELS, current_model
    if not api_key:
        print("Error: No API key set for fetching models.")
        return False
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        for model in models:
            if model["id"] == model_id:
                model_key = model_id.split(":")[0] if ":" in model_id else model_id.split("/")[-1]
                MODELS[model_key] = model_id
                print(f"Model '{model_id}' added successfully as '{model_key}'.")
                if set_as_current:
                    current_model = model_key
                    print(f"Current model set to: {model_key}")
                return True
        print(f"Error: Model '{model_id}' not found in OpenRouter's model list.")
        return False
    except Exception as e:
        print(f"Error fetching model list: {e}")
        return False

def send_prompt(prompt: str, history: InMemoryHistory) -> Optional[str]:
    """Send prompt to the current model and stream response."""
    global conversation_history
    if not current_model:
        print("Error: No model selected. Use /set <model-name> to select a model.")
        return None
    if not client and not initialize_client():
        return None
    try:
        # Prepare messages
        messages = []
        if chat_mode:
            messages.extend(conversation_history)
        else:
            messages = []

        # Add system message for reasoning mode (fallback for non-reasoning-token models)
        if reasoning_mode:
            messages.append({
                "role": "system",
                "content": "You are an expert reasoner. Break down the problem step by step within `<think>` tags, showing your reasoning process clearly. Provide the final answer outside the `<think>` tags."
            })

        # Use original prompt for reasoning token models, modified prompt for others
        user_content = (
            f"<think>Let's reason through this step by step: {prompt}</think>"
            if not reasoning_mode or current_model not in ["deepseek-r1", "claude-3.7-sonnet"]  # Add more reasoning-token models as needed
            else prompt
        )
        messages.append({"role": "user", "content": user_content})

        # Configure reasoning parameters
        extra_params = {}
        if reasoning_mode:
            extra_params["reasoning"] = {"effort": reasoning_effort}

        stream = client.chat.completions.create(
            model=MODELS[current_model],
            messages=messages,
            max_tokens=2000 if reasoning_mode else 1000,
            temperature=0.5 if reasoning_mode else 1.0,
            stream=True,
            extra_body=extra_params
        )
        
        print("\nResponse:")
        response_text = ""
        reasoning_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning") and delta.reasoning is not None:
                reasoning_content = delta.reasoning
                print(f"<think>{reasoning_content}</think>", end="", flush=True)
                reasoning_text += reasoning_content
            elif delta.content is not None:
                content = delta.content
                print(content, end="", flush=True)
                response_text += content

        print("\n")  # After streaming

        if pretty_markdown and (response_text.strip() or reasoning_text.strip()):
            print("\nFormatted Output:\n")
            if reasoning_text:
                console.print(Markdown(f"**Reasoning:**\n<think>{reasoning_text}</think>"))
            if response_text:
                console.print(Markdown(f"**Answer:**\n{response_text}"))

        # Update conversation history if in chat mode
        if chat_mode:
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": response_text})

        history.append_string(prompt)  # Add prompt to history
        return response_text
    except KeyboardInterrupt:
        print("\nStream interrupted by Ctrl+C.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def handle_command(input_str: str, history: InMemoryHistory) -> bool:
    """Handle configuration commands. Returns True if a command was processed."""
    global api_key, current_model, MODELS, pretty_markdown, chat_mode, reasoning_mode, reasoning_effort, conversation_history
    parts = input_str.strip().split(maxsplit=2)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    extra_args = parts[2] if len(parts) > 2 else ""

    if command == "/key":
        if not args:
            print("Error: Please provide an API key. Usage: /key <your-api-key>")
            return True
        api_key = args
        if initialize_client():
            print("API key updated successfully.")
        return True
    if command == "/clear":
        clear_screen()
        return True
    if command == "/models":
        print("Available models:")
        for model in MODELS:
            print(f"- {model}{' (current)' if model == current_model else ''}")
        return True
    if command == "/set":
        if not args:
            print("Error: Please provide a model name. Usage: /set <model-name>")
            return True
        model_name = args.lower()
        if model_name not in MODELS:
            print(f"Error: Invalid model. Available models: {', '.join(MODELS.keys())}")
            return True
        current_model = model_name
        print(f"Current model set to: {current_model}")
        return True
    if command == "/free":
        free_models = fetch_free_models()
        if free_models:
            MODELS.update(free_models)
            print("Free models available on OpenRouter:")
            for name, id in free_models.items():
                print(f"- {name} ({id})")
            print("These models have been added to available models. Use /set <model-name> to select.")
        else:
            print("No free models fetched. Check API key or network.")
        return True
    if command == "/pretty":
        if not args:
            print(f"Pretty Markdown rendering is currently {'ON' if pretty_markdown else 'OFF'}. Use /pretty on|off to change.")
            return True
        if args.lower() == "on":
            pretty_markdown = True
            print("Pretty Markdown rendering ENABLED.")
        elif args.lower() == "off":
            pretty_markdown = False
            print("Pretty Markdown rendering DISABLED.")
        else:
            print("Error: Usage: /pretty on|off")
        return True
    if command == "/chat":
        if not args:
            print(f"Chat mode is currently {'ON' if chat_mode else 'OFF'}. Use /chat on|off to change.")
            return True
        if args.lower() == "on":
            chat_mode = True
            conversation_history = []
            print("Chat mode ENABLED. Conversation history will be maintained.")
        elif args.lower() == "off":
            chat_mode = False
            conversation_history = []
            print("Chat mode DISABLED. Conversation history cleared.")
        else:
            print("Error: Usage: /chat on|off")
        return True
    if command == "/reasoning":
        if not args:
            print(f"Reasoning mode is currently {'ON' if reasoning_mode else 'OFF'}. Use /reasoning on|off to change.")
            return True
        if args.lower() == "on":
            reasoning_mode = True
            print("Reasoning mode ENABLED. Responses will include step-by-step reasoning.")
        elif args.lower() == "off":
            reasoning_mode = False
            print("Reasoning mode DISABLED.")
        else:
            print("Error: Usage: /reasoning on|off")
        return True
    if command == "/effort":
        if not args:
            print(f"Reasoning effort is currently '{reasoning_effort}'. Use /effort low|medium|high to change.")
            return True
        if args.lower() in ["low", "medium", "high"]:
            reasoning_effort = args.lower()
            print(f"Reasoning effort set to: {reasoning_effort}")
        else:
            print("Error: Usage: /effort low|medium|high")
        return True
    if command == "/addmodel":
        if not args:
            print("Error: Please provide a model ID. Usage: /addmodel <model-id> [set]")
            return True
        set_as_current = extra_args.lower() == "set"
        if add_non_free_model(args, set_as_current):
            print(f"Model '{args}' is now available. Use /models to see all models or /set {args.split(':')[0]} to select it.")
        return True
    if command == "/help":
        print("Available commands:")
        print("/key <your-api-key>  - Set OpenRouter API key")
        print("/models             - List available models")
        print("/set <model-name>   - Set the current model")
        print("/free               - Fetch and list free models")
        print("/addmodel <model-id> [set] - Add a model by ID and optionally set as current")
        print("/pretty on|off      - Toggle pretty Markdown rendering")
        print("/chat on|off        - Toggle conversation mode")
        print("/reasoning on|off   - Toggle reasoning mode with Reasoning Tokens")
        print("/effort low|medium|high - Set reasoning effort level")
        print("/help               - Show this help message")
        print("Use Ctrl+Enter for newlines, Alt+Enter to submit prompt, or 'exit' to quit.")
        print("Press Up/Down arrows to cycle through prompt history, Ctrl+C to stop streaming.")
        print("Note: If keybindings don't work, try a different terminal (e.g., VS Code, iTerm2).")
        return True
    return False

def create_prompt_session() -> PromptSession:
    """Create a PromptSession with custom key bindings and history."""
    bindings = KeyBindings()
    history = InMemoryHistory()

    @bindings.add('enter')  # Alt+Enter for submit
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    @bindings.add('c-x')  # Ctrl+Enter for newline
    def _(event):
        event.app.current_buffer.insert_text('\n')

    @bindings.add('up')  # Up arrow for previous prompt
    def _(event):
        buffer = event.app.current_buffer
        if history.get_strings() and buffer.text == "":
            buffer.text = history.get_strings()[-1]
            buffer.cursor_position = len(buffer.text)

    @bindings.add('down')  # Down arrow for next prompt
    def _(event):
        buffer = event.app.current_buffer
        buffer.reset()

    return PromptSession(
        multiline=True,
        prompt_continuation='... ',
        key_bindings=bindings,
        history=history,
        complete_while_typing=False
    )

def main():
    print("Interactive OpenRouter Chat Script")
    print("==================================")
    print("Type /help for available commands.")
    print("Use arrow keys to navigate, Ctrl+Enter for newlines, Alt+Enter to submit.")
    
    MODELS.update(fetch_free_models() or {})

    if api_key:
        if initialize_client():
            print("API key loaded successfully from .env.")
        else:
            print("Failed to initialize with .env API key. Use /key to set manually.")

    session = create_prompt_session()

    while True:
        try:
            model_display = current_model if current_model else 'no model'
            pretty_display = "pretty" if pretty_markdown else "raw"
            chat_display = "chat" if chat_mode else "single"
            reasoning_display = f"reasoning ({reasoning_effort})" if reasoning_mode else "normal"
            user_input = session.prompt(f"\n[{model_display} | {pretty_display} | {chat_display} | {reasoning_display}] > ")
        except KeyboardInterrupt:
            print("\nInput interrupted. Continue or type 'exit' to quit.")
            continue

        user_input = user_input.strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        if user_input.startswith("/") and handle_command(user_input, session.history):
            continue

        if not current_model:
            print("Error: No model selected. Use /set <model-name> to select a model.")
            continue
        response = send_prompt(user_input, session.history)
        if not response:
            print("Failed to get response. Check API key and model settings.")

if __name__ == "__main__":
    main()