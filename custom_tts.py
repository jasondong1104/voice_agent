from livekit.plugins.elevenlabs.tts import TTS as ElevenLabsTTS
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

class CustomElevenLabsTTS(ElevenLabsTTS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._conn_options = DEFAULT_API_CONNECT_OPTIONS

    def stream(
        self, 
        *, 
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs
    ):
        self._conn_options = conn_options
        return super().stream(conn_options=conn_options, **kwargs)
