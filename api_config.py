from dataclasses import dataclass
from typing import Dict
DEFAULT_AGENT_PROFILE = """""
        个人资料：你是一位新入职山西太原市杨帆高中的物理老师；最近几个月在负责学校的招生工作，正在与学生家长进行电话联系。
        谈话策略：
        1.在通话话开始阶段首先介绍自己。
        2.然后主动了解对方的家庭情况，孩子的升学情况。通话过程中要始终记得你的目的是为杨帆高中招生。
        3.在谈话过程中保持主动！每次对话都要注意跟通话目的和策略保持一致。
        谈话的目的：
        1.了解对方的家庭情况，特别是对方的孩子的升学情况。
        2.如果对方的家庭中有孩子准备升学进入高中阶段，就向其推荐杨帆中学，并邀请其加入群聊；
        3.对方表达意愿后可逐渐结束对话。如果对方没有准备升学的孩子，逐渐结束对话。
        约束：
        1.回答使用口语化的语言，并保持简洁。
        2.讲话的声音不要太慢，回答不要超过50个字。
        3.回答具体问题时精炼一些，少用感叹句。
        4.对于你不懂的问题，回答你自己不了解；或者邀请对方加入群聊向骨干老师咨询。
        杨帆高中资料：
        1.对学生的激励措施：高中期间的奖金和竞赛辅导。
        2.免费的公益活动：对有升学意向的初三学生提供免费的体育加试辅导和培训。
        """
@dataclass
class APIConfig:
    profile: str = DEFAULT_AGENT_PROFILE
    stt_model: str = "deepgram"
    llm_model: str = "grok-2-latest"
    tts_model: str = "elevenlabs"
    stt_api_key: str = ""
    llm_api_key: str = ""
    tts_api_key: str = ""
    stt_base_url: str = ""
    llm_base_url: str = ""
    tts_base_url: str = ""

    @staticmethod
    def from_dict(data: Dict[str, str]) -> 'APIConfig':
        return APIConfig(
            stt_model=data.get('stt_model', "deepgram"),
            llm_model=data.get('llm_model', "grok-2-latest"),
            tts_model=data.get('tts_model', "elevenlabs"),
            stt_api_key=data.get('stt_api_key', ""),
            llm_api_key=data.get('llm_api_key', ""),
            tts_api_key=data.get('tts_api_key', ""),
            stt_base_url=data.get('stt_base_url', ""),
            llm_base_url=data.get('llm_base_url', ""),
            tts_base_url=data.get('tts_base_url', "")
        )

config = APIConfig()

def update_config(new_config: Dict[str, str]):
    global config
    config = APIConfig.from_dict(new_config)
    print(f"updated config: {config}")

# 添加一个方法来获取默认配置
def get_default_config() -> Dict[str, str]:
    return {
        'profile': DEFAULT_AGENT_PROFILE,
        'stt_model': 'deepgram',
        'llm_model': 'grok-2-latest',
        'tts_model': 'elevenlabs',
        'stt_api_key': '',
        'llm_api_key': '',
        'tts_api_key': '',
        'stt_base_url': '',
        'llm_base_url': '',
        'tts_base_url': ''
    }
