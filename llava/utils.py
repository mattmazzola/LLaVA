from enum import Enum
import json
import logging
import logging.handlers
import os
import sys

from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, ImageData

import requests
from .guardlistWrapper import GuardlistWrapper

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_input_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
moderation_output_msg = "Sorry, the model output content which violates our content moderation policies."


class ModerationOptions(Enum):
    ALL: str = "all"

    INPUT_TEXT_GUARDLIST: str = "input_text_guardlist"
    INPUT_TEXT_AICS: str = "input_text_aics"
    INPUT_TEXT_OPENAI: str = "input_text_openai"
    INPUT_IMAGE_AICS: str = "input_image_aics"

    OUTPUT_TEXT_GUARDLIST: str = "output_text_guardlist"
    OUTPUT_TEXT_AICS: str = "output_text_aics"
    OUTPUT_TEXT_OPENAI: str = "output_text_openai"

    GLIGEN_INPUT_TEXT_GUARDLIST: str = "gligen_input_text_guardlist"
    GLIGEN_INPUT_TEXT_AICS: str = "gligen_input_text_aics"
    GLIGEN_OUTPUT_IMAGE_AICS: str = "gligen_output_image_aics"


handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", logging.INFO))
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


logger = build_logger("llava-utils", "llava-utils.log")


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


GuardlistWrapper.appKey = os.environ["GUARDLIST_KEY"]
GuardlistWrapper.partnerName = "LLaVA-Interactive-Demo"


def violates_guardlist_moderation(text):
    is_phrase_problematic = GuardlistWrapper.is_phrase_problematic(text, "en")

    return is_phrase_problematic


def violates_openai_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    text = text.replace("\n", "")
    data = {"input": text}

    try:
        ret = requests.post(url, headers=headers, data=json.dumps(data).encode("utf-8"), timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


client: ContentSafetyClient = None


def does_text_violate_azure_content_safety(text,
                                           hate_severity_max=3,
                                           self_harm_severity_max=3,
                                           sexual_severity_max=3,
                                           violence_severity_max=3):
    does_violate = False

    global client
    if client is None:
        endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
        key = os.environ["CONTENT_SAFETY_KEY"]
        client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    try:
        request = AnalyzeTextOptions(text=text)
        response = client.analyze_text(request)

        category_results_vs_max_severity_pairs = [
            (response.hate_result, hate_severity_max),
            (response.self_harm_result, self_harm_severity_max),
            (response.sexual_result, sexual_severity_max),
            (response.violence_result, violence_severity_max),
        ]
        logger.debug(f"Text '{text}' content severity: {', '.join([f'{s.category}: {s.severity}' for s, _ in category_results_vs_max_severity_pairs])}")

        for category_result, max_severity in category_results_vs_max_severity_pairs:
            if category_result is not None:
                if category_result.severity > max_severity:
                    does_violate = True
                    logger.warning(f"⚠️ '{text}' violated the {category_result.category} policy. Actual Severity: {category_result.severity} > Max Severity: {max_severity}")
                    break

    except HttpResponseError as e:
        logger.error("Analyze text failed.")
        if e.error:
            logger.error(f"Error code: {e.error.code}")
            logger.error(f"Error message: {e.error.message}")

    return does_violate


def _convert_image_base64_str(image):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_b64_str


def does_image_violate_azure_content_safety(image,
                                            hate_severity_max=3,
                                            self_harm_severity_max=3,
                                            sexual_severity_max=3,
                                            violence_severity_max=3):
    does_violate = False

    global client
    if client is None:
        endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
        key = os.environ["CONTENT_SAFETY_KEY"]
        client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    try:
        image_base64_str = _convert_image_base64_str(image)
        request = AnalyzeImageOptions(image=ImageData(content=image_base64_str))
        response = client.analyze_image(request)

        category_results_vs_max_severity_pairs = [
            (response.hate_result, hate_severity_max),
            (response.self_harm_result, self_harm_severity_max),
            (response.sexual_result, sexual_severity_max),
            (response.violence_result, violence_severity_max),
        ]
        logger.debug(f"Image content severity: {', '.join([f'{s.category}: {s.severity}' for s, _ in category_results_vs_max_severity_pairs])}")

        for category_result, max_severity in category_results_vs_max_severity_pairs:
            if category_result is not None:
                if category_result.severity > max_severity:
                    does_violate = True
                    logger.warning(f"⚠️ The image provided violated the {category_result.category} policy. Actual Severity: {category_result.severity} > Max Severity: {max_severity}")
                    break

    except HttpResponseError as e:
        logger.error("Analyze image failed.")
        if e.error:
            logger.error(f"Error code: {e.error.code}")
            logger.error(f"Error message: {e.error.message}")

    return does_violate


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
