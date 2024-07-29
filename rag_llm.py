import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from document_loader import DocumentStore
from language_model import LanguageModel, OpenAILanguageModel
from prompt_format import NEW_AGENT_PROMPT, PROMPT_FORMAT

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGLanguageModel:
    def __init__(
        self,
        api_key: str,
        document_folder: Path | str,
        llm: LanguageModel,
        document_store: DocumentStore,
    ):
        self._api_key = api_key
        self._path = document_folder
        self._llm = llm
        self._agent = self._llm(api_key=self._api_key)
        self._document_store = document_store(path=self._path, api_key=self._api_key)
        self._document_store.load_documents_to_index()

    def start_conversation(self) -> str:
        while True:
            question = input("Input: ")
            if question.lower() == "q":
                break

            if len(self._agent.get_message_history()) > 1:
                new_question = self._add_context_to_question(question)
            else:
                new_question = question
            context = self._document_store.search(new_question)
            formatted_context = self._format_context(context)
            prompt = PROMPT_FORMAT.format(formatted_context, new_question)

            full_response = ""
            response = self._agent.chat(prompt=prompt, stream=True)
            for chunk in response:
                if chunk:
                    full_response += chunk
                    print(chunk, end="")
                elif chunk is False:
                    print("\n")
                    break
            self._change_message_history(new_question, full_response)

    def _format_context(self, context: list) -> str:
        formatted_context = ""
        for index, document in enumerate(context):
            formatted_context += f"Document {index + 1})\n"
            formatted_context += document.page_content
            formatted_context += "\n\n"

        return formatted_context

    def _add_context_to_question(self, question: str) -> str:
        new_agent = self._create_agent()
        message_history = self._agent.get_message_history()
        formatted_message_history = self._format_message_history(message_history)
        agent_prompt = NEW_AGENT_PROMPT.replace("_USER_QUESTION_", question)
        agent_prompt = agent_prompt.replace(
            "_MESSAGE_HISTORY_", formatted_message_history
        )
        response = new_agent.chat(prompt=agent_prompt, stream=True)
        new_question = ""
        for chunk in response:
            if chunk:
                new_question += chunk
            elif chunk is False:
                break

        return new_question

    def _format_message_history(self, message_history: list[dict]) -> str:
        formatted_message_history = ""
        for message in message_history:
            role = message["role"]
            content = message["content"]
            formatted_message_history += f"{role}: {content}\n"

        return formatted_message_history

    def _change_message_history(self, user_question: str, ai_response: str) -> str:
        current_messages = self._agent.get_message_history()
        latest_user_message = current_messages[-2]
        latest_ai_message = current_messages[-1]

        latest_user_message["content"] = user_question
        latest_ai_message["content"] = ai_response

        current_messages[-2] = latest_user_message
        current_messages[-1] = latest_ai_message

        self._agent.set_message_history(current_messages)

    def _create_agent(self) -> OpenAILanguageModel:
        new_agent = self._llm(api_key=self._api_key, sys_prompt=False)
        return new_agent


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    load_dotenv()

    folder = Path("./docs")
    API_KEY = os.environ.get("OPEN_AI_KEY")

    rag = RAGLanguageModel(
        api_key=API_KEY,
        document_folder=folder,
        llm=OpenAILanguageModel,
        document_store=DocumentStore,
    )
    rag.start_conversation()
