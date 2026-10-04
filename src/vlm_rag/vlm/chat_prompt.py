"""One prompt contract shared by supervised training and live chat inference."""

import json

from vlm_rag.vlm.models import VLMTaskType

CHAT_PROMPT_VERSION = "semantic-evidence-chat-v1"


def task_instruction(task: VLMTaskType, request_id: str) -> str:
    if task in {VLMTaskType.OCR_RECOVERY, VLMTaskType.REGION_TRANSCRIPTION}:
        payload: dict[str, object] = {"transcription": "exact visible text"}
    elif task in {VLMTaskType.TABLE_TEXT_RECOVERY, VLMTaskType.TABLE_HEADER_RECOVERY}:
        payload = {"table_rows": [["exact cell text"]]}
    else:
        payload = {"description": "visible observation only"}
    return "Return this JSON shape, replacing only the payload: " + json.dumps(
        {
            "schema_version": 1,
            "request_id": request_id,
            "task_type": task.value,
            **payload,
        }
    )
