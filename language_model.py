from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI

from prompt_format import SYS_PROMPT

load_dotenv()


class LanguageModel(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass

    @abstractmethod
    def reset(self, sys_prompt: str = None) -> None:
        pass


class OpenAILanguageModel(LanguageModel):
    def __init__(self, api_key: str, sys_prompt: str | bool = None):
        self.client = OpenAI(api_key=api_key)
        if sys_prompt is not False:
            sys_prompt = SYS_PROMPT if not sys_prompt else sys_prompt
            self._messages = [
                {"role": "system", "content": SYS_PROMPT},
            ]
        else:
            self._messages = []

    def chat(self, prompt: str, stream: bool = False):
        self._messages.append({"role": "user", "content": prompt})
        write_messages_to_file(self._messages)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._messages,
            stream=stream,
        )
        if not stream:
            response = response.choices[0].message.content
            self._messages.append({"role": "assistant", "content": response})

            return response
        else:
            full_message = ""
            for chunk in response:
                response_chunk = chunk.choices[0].delta.content
                full_message += response_chunk if response_chunk else ""
                yield response_chunk

            self._messages.append({"role": "assistant", "content": full_message})
            write_messages_to_file(self._messages)
            yield False

    def set_message_history(self, messages: list[dict]):
        self._messages = messages
        write_messages_to_file(self._messages)

    def get_message_history(self) -> list[dict]:
        return self._messages

    def reset(self, sys_prompt: str = None) -> None:
        if not sys_prompt:
            sys_prompt = SYS_PROMPT
        self._messages = [
            {"role": "system", "content": sys_prompt},
        ]


def write_messages_to_file(messages: list[dict]):
    with open("message_history.txt", "w") as file:
        for message in messages:
            file.write(f"{message['role']}: {message['content']}\n\n")
