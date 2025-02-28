from dataclasses import dataclass
from typing import Dict

@dataclass
class APIConfig:
    stt_model: str = "whisper"
    llm_model: str = "gpt-3.5-turbo"
    tts_model: str = "elevenlabs"
    stt_api_key: str = ""
    llm_api_key: str = ""
    tts_api_key: str = ""

    @staticmethod
    def from_dict(data: Dict[str, str]) -> 'APIConfig':
        return APIConfig(
            stt_model=data.get('stt_model', "whisper"),
            llm_model=data.get('llm_model', "gpt-3.5-turbo"),
            tts_model=data.get('tts_model', "elevenlabs"),
            stt_api_key=data.get('stt_api_key', ""),
            llm_api_key=data.get('llm_api_key', ""),
            tts_api_key=data.get('tts_api_key', "")
        )

config = APIConfig()

def update_config(new_config: Dict[str, str]):
    global config
    config = APIConfig.from_dict(new_config)
