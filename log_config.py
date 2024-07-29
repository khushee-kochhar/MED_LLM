import logging

logger = logging.getLogger("RAG_LLM")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "Level:=%(levelname)s::Timestamp:=%(asctime)s::Msg:=%(message)s"
)
formatter = logging.Formatter("%(message)s")

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.propagate = False


def get_logger():
    return logger
