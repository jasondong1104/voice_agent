import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import langchain_plugin as lc
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from typing import Any, Dict, List
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

class MyMessagesState(MessagesState):
    records: str

llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
# ...
teacher_profile = """个人资料：你是一位新入职山西太原市杨帆高中的物理老师；最近几个月在负责学校的招生工作，正在与学生家长进行电话联系。
谈话策略：在通话话开始阶段首先介绍自己。然后主动了解对方的家庭情况，孩子的升学情况。通话过程中要始终记得你的目的是为杨帆高中招生。在谈话过程中保持主动！每次对话都要注意跟通话目的和策略保持一致。
谈话的目的是了解对方的家庭情况，特别是对方的孩子的升学情况。如果对方的家庭中有孩子准备升学进入高中阶段，就向其推荐杨帆中学，并邀请其加入群聊；对方表达意愿后可逐渐结束对话。如果对方没有准备升学的孩子，逐渐结束对话。
约束：
回答使用口语化的语言，并保持简洁。
讲话的声音不要太慢，回答不要超过50个字。
回答具体问题时精炼一些，少用感叹句。
对于你不懂的问题，回答你自己不了解；并且邀请对方加入群聊向骨干老师咨询。
杨帆高中资料：
对学生的激励措施：高中期间的奖金和竞赛辅导。
免费的公益活动：
对有升学意向的初三学生提供免费的体育加试辅导和培训。
免费参观学校的活动。"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", teacher_profile),
        ("placeholder", "{messages}"),
    ]
)
prompt_end = ChatPromptTemplate.from_messages(
    [
        ("system", teacher_profile + 
         '\n请根据聊天记录和修改意见，尽量改善你准备对用户做出的回答，输出最终决定回答的内容。'+
        "\n聊天记录：{messages}\n 修改意见：{fix}\n 之前准备做出的回答：{response}\n 改善后的回答：" ), 
    ]
)
prompt_gate = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个负责招生的老师，请根据以下与用户的聊天记录，判断用户最终是否同意加入招生群."),
        ("placeholder", "{messages}"),
    ]
)
prompt_fix = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个善于为学校招生并且具有良好的沟通能力的老师，"
         "请根据以下与用户的聊天记录和你的职业特点判断后续的这个回答是否合适，并且提出改善的意见."
         "\n聊天记录：{messages}\n 职业特点：{profile}\n 回答：{response}\n "),
    ]
)
agent_gate = prompt_gate | llm.bind(temperature=0) | BooleanOutputParser()
agent_continue = prompt | llm| StrOutputParser()
agent_fix = prompt_fix | llm | StrOutputParser()
agent_out = prompt_end | llm | StrOutputParser()

# Convert node functions to async
async def node_continue(ctx: MyMessagesState) -> Dict[str, Any]:
    # Convert messages to Langchain format
    messages = ctx["messages"]
    ctx['records'] = ''
    for msg in messages:
        ctx['records'] += (('user: ' if isinstance(msg, HumanMessage)  else 'assistant: ') + msg.content + '\n')
        print('record is ', ctx['records'])
    #print('first node get messages: ', ctx['records'])

    print('first node get messages: ', messages)
    # Use ainvoke for async operation
    response = await agent_continue.ainvoke({"messages": ctx["messages"]})
    #print(f'ctx is {ctx}, response is {response}')
    return {'messages': [('assistant', response)], 'records': ctx['records']}

async def node_fix(ctx: MyMessagesState):
    #print('first response is ', ctx["messages"][-1])
    print('fix nod get ctx: ', ctx)
    rsp1 = await agent_fix.ainvoke({
        'messages': ctx["records"], 
        'profile': teacher_profile, 
        'response': ctx["messages"][-1].content,
    })
    #print('llm fix answer with suggestion: ', rsp1)
    return {'messages': [('assistant', rsp1)]}

async def node_out(ctx: MyMessagesState):
    rsp = await agent_out.ainvoke({
        'messages': ctx["records"], 
        'fix': ctx["messages"][-1].content, 
        'response': ctx["messages"][-2].content
    })
    #print('llm final answer: ', rsp)
    ctx["messages"] = ctx["messages"][:-2]
    return {'messages': [('assistant', rsp)]}

# Create async workflow
workflow = StateGraph(MyMessagesState)

# Add nodes with async functions
workflow.add_node("node1", node_continue)
workflow.add_node("node2", node_fix)
workflow.add_node("node3", node_out)

workflow.set_entry_point('node1')
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", "node3")
workflow.add_edge("node3", END)

llm_workflow = workflow.compile()

# Test code
if __name__ == '__main__':
    async def test_workflow():
        msgs = {'messages': [
            #{'role': 'system', 'content': '你是一位历史学家。'}, 
            HumanMessage(content='介绍一下川普')
            ]
        }
        rst = await llm_workflow.ainvoke(msgs)
        print(rst)
    asyncio.run(test_workflow())
