import os
from typing import List, Dict, Any, Callable, Optional
from IPython.display import display, Markdown

from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.markdown import Markdown

load_dotenv(override=True)

# Optional imports
try:
    import ollama
except ImportError:
    ollama = None

try:
    import openai
except ImportError:
    openai = None

try:
    import google.genai as genai
except ImportError:
    genai = None


class UnifiedLLM:
    """
    Single entry-point for multiple cloud LLM providers.
    """

    PROVIDERS = {"ollama", "groq", "hf", "openrouter", "google", "github"}

    # Type hint for the ask function: (client, model, prompt, **kwargs) -> str
    AskFn = Callable[[Any, str, str, Dict[str, Any]], str]

    _cfg: Dict[str, Dict[str, Any]] = {
        "ollama": {
            "client_cls": lambda: ollama.Client(
                host="https://ollama.com",
                headers={"Authorization": f'Bearer {os.getenv("OLLAMA_API_KEY")}'},
            ),
            # Added **kwargs support here
            "ask_fn": lambda client, model, prompt, **kwargs: client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                **kwargs,
            )["message"]["content"],
            "default_model": "gpt-oss:120b-cloud",
            "list_fn": lambda client: [m.model for m in client.list().models],
        },
        "groq": {
            "client_cls": lambda: openai.OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            ),
            # Groq uses 'responses' endpoint in your original code
            "ask_fn": lambda client, model, prompt, **kwargs: client.responses.create(
                input=prompt,
                model=model,
                **kwargs,  # Passed kwargs might be ignored by SDK if invalid, but we offer the option
            ).output_text,
            "default_model": "qwen/qwen3-32b",
            "list_fn": lambda client: [m.id for m in client.models.list()],
        },
        "hf": {
            "client_cls": lambda: openai.OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.getenv("HF_TOKEN"),
            ),
            "ask_fn": lambda client, model, prompt, **kwargs: client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], **kwargs
            )
            .choices[0]
            .message.content,
            "default_model": "zai-org/GLM-4.7:novita",
            "list_fn": lambda client: [m.id for m in client.models.list()],
        },
        "openrouter": {
            "client_cls": lambda: openai.OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            ),
            "ask_fn": lambda client, model, prompt, **kwargs: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
            .choices[0]
            .message.content,
            "default_model": "google/gemma-3-27b-it:free",
            "list_fn": lambda client: [
                m.id for m in client.models.list() if "free" in m.id
            ],
        },
        "google": {
            # Create the client instance
            "client_cls": lambda: genai.Client(
                api_key=os.getenv("GOOGLEAISTUDIO_API_KEY")
            ),
            # Define how to ask
            "ask_fn": lambda client, model, prompt, **kwargs: client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.GenerateContentConfig(**kwargs) if kwargs else None,
            ).text,
            "default_model": "models/gemini-2.5-flash",
            # Define how to list models
            "list_fn": lambda client: [m.name for m in client.models.list()],
        },
        "github": {
            "client_cls": lambda: openai.OpenAI(
                api_key=os.getenv("GITHUB_API_KEY"),
                base_url="https://models.inference.ai.azure.com",
            ),
            # Removed hardcoded temp/max_tokens to allow flexibility via ask(**kwargs)
            "ask_fn": lambda client, model, prompt, **kwargs: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
            .choices[0]
            .message.content,
            "default_model": "gpt-4o-mini",
            "list_fn": lambda client: [m.id for m in client.models.list()],
        },
    }

    def __init__(self, provider: str, model: str = None):
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Pick from {self.PROVIDERS}"
            )

        self.provider = provider
        self.model = model or self._cfg[provider]["default_model"]

        # Initialize client
        try:
            self._raw_client = self._cfg[provider]["client_cls"]()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize client for {provider}: {e}")

    def ask(self, prompt: str, **kwargs) -> str:
        """Return the model's answer as plain text."""
        try:
            fn = self._cfg[self.provider]["ask_fn"]
            # Pass kwargs to the specific provider function
            return fn(self._raw_client, self.model, prompt, **kwargs)
        except Exception as e:
            # Optionally wrap exceptions to provide context
            print(f"Error during request to {self.provider}: {e}")
            raise

    def ask_md(self, prompt: str, **kwargs) -> None:
        """Pretty-print the answer as Markdown (Jupyter-friendly)."""
        return self.ask(prompt, **kwargs)

    def list_models(self) -> List[str]:
        """Return a list of available models for this provider."""
        try:
            return self._cfg[self.provider]["list_fn"](self._raw_client)
        except Exception as e:
            print(f"Error listing models for {self.provider}: {e}")
            return []

    def set_model(self, model: str) -> None:
        """Switch to a different model for the same provider."""
        self.model = model


# ----------------------------------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------------------------------
def main() -> None:
    """The main of program"""

    # Groq
    print("\n" + "=" * 50)
    print("Example 1: Basic Usage with Groq")
    print("=" * 50)
    try:
        groq = UnifiedLLM("groq")
        print(f"Using Model: {groq.model}")
        response = groq.ask("What is the capital of Iran?")
        Console().print(Markdown(f"Response: {response}"))
    except Exception as e:
        print(f"Skipping Groq example: {e}")

    # Google AI Studio
    print("\n" + "=" * 50)
    print("Example 2: Basic Usage with Google AI Studio")
    print("=" * 50)
    try:
        google = UnifiedLLM("google")
        print(f"Using Model: {google.model}")
        response = google.ask("What is the capital of Iran?")
        Console().print(Markdown(f"Response: {response}"))

        print(google.list_models()[:5])
    except Exception as e:
        print(f"Skipping Google AI Studio example: {e}")

    # OpenRouter
    print("\n" + "=" * 50)
    print("Example 3: Markdown Output & Custom Model (OpenRouter)")
    print("=" * 50)
    try:
        or_llm = UnifiedLLM("openrouter")
        # Switch to a specific model programmatically
        or_llm.set_model("nex-agi/deepseek-v3.1-nex-n1:free")
        print(f"Switched Model to: {or_llm.model}")

        print("Outputting Markdown...")
        Console().print(
            Markdown(
                or_llm.ask_md("Write a short Python function to reverse a string.")
            )
        )
    except Exception as e:
        print(f"Skipping OpenRouter example: {e}")

    # Github
    print("\n" + "=" * 50)
    print("Example 4: Using Parameters (GitHub)")
    print("=" * 50)
    try:
        github = UnifiedLLM("github")
        # Pass parameters like temperature or max_tokens
        response = github.ask(
            "Generate a random number between 1 and 100.",
            temperature=1.0,  # High randomness
            max_tokens=50,
        )
        Console().print(Markdown(f"GitHub Response: {response}"))
    except Exception as e:
        print(f"Skipping GitHub example: {e}")

    # Ollama
    print("\n" + "=" * 50)
    print("Example 5: Listing Models (Ollama)")
    print("=" * 50)
    try:
        ollama_client = UnifiedLLM("ollama")
        models = ollama_client.list_models()
        print(f"Total models found: {len(models)}")
        print("First 3 models:", models[:3])
    except Exception as e:
        print(f"Skipping Ollama example: {e}")

    # HuggingFace
    print("\n" + "=" * 50)
    print("Example 6: HuggingFace Standard Chat")
    print("=" * 50)
    try:
        hf = UnifiedLLM("hf")
        print(f"Model: {hf.model}")
        Console().print(
            Markdown(hf.ask_md("Explain quantum computing to a 5-year-old."))
        )
    except Exception as e:
        print(f"Skipping HuggingFace example: {e}")


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Note: Ensure your .env file is populated with the necessary API keys before running.
    os.system(command="cls" if os.name == "nt" else "clear")

    try:
        main()
    except KeyboardInterrupt:
        print()
    except Exception as error:
        print(f"[-] {error}")

    print()
