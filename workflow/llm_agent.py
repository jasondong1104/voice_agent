import asyncio
import os
from pprint import pprint
from dotenv import load_dotenv
# print(load_dotenv('.env.local'))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from typing import Any, Dict, List
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from backend.api_config import config

from workflow.rag import rag_query

TOOLS = [rag_query]

class MyMessagesState(MessagesState):
    records: str
    rag_data: str
    first_rsp: str

class WorkFlow:

    def __init__(self):
        self.llm = self.get_llm()
        self.teacher_profile = """个人资料：你是一位新入职山西太原市杨帆高中的物理老师；最近几个月在负责学校的招生工作，正在与学生家长进行电话联系。
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
        对有升学意向的初三学生提供免费的体育加试辅导和培训。"""
        self.init_agents()
        self.node_tool = ToolNode(TOOLS)

    @staticmethod
    def get_llm():
        return ChatOpenAI(
            temperature=0.5,
            model=config.llm_model if config.llm_model else "grok-2-latest",
            api_key=config.llm_api_key if config.llm_api_key else os.getenv("OPENAI_API_KEY")
        )
    @staticmethod
    def continue_or_end(ctx: MyMessagesState):
        if ctx['messages'][-1].tool_calls:
            print('goto tool')
            return "tool_node"
        else:
            print('goto node1')
            return 'node1'
        
    async def node_gate(self, ctx: MyMessagesState) -> Dict[str, Any]:
        # Convert messages to Langchain format
        messages = ctx["messages"]
        ctx['records'] = ''
        for msg in messages:
            ctx['records'] += (('user: ' if isinstance(msg, HumanMessage)  else 'assistant: ') + msg.content + '\n')
            print('record is ', ctx['records'])
        #print('gate node get records: ', ctx['records'])

        print('gate node get messages: ', messages)
        # Use ainvoke for async operation
        print('gate node start ...')
        response = await self.agent_gate.ainvoke({"messages": ctx["records"]})
        print('gate node generated rsp.')
        pprint(f'gate node rsp is: {response}')
        return {'messages': [response], 'records': ctx['records']}
    
    async def node_continue(self, ctx: MyMessagesState) -> Dict[str, Any]:
        print('node-continue get ctx:\n', ctx)
        if hasattr(ctx["messages"][-2],'tool_calls'):
            rag_data = ctx["messages"][-1].content
            print('rag data is:\n', rag_data)
        else:
            rag_data = '(无)'

        # Use ainvoke for async operation
        response = await self.agent_continue.ainvoke({"messages": ctx["messages"], 'rag_data': rag_data})
        first_rsp = response
        print(f'first response  content is: {first_rsp}')
        return {'messages': [('assistant', response)], 
                'first_rsp': first_rsp, 
                'rag_data': rag_data
                }
    
    
    async def node_fix(self, ctx: MyMessagesState):
        #print('first response is ', ctx["messages"][-1])
        print('fix nod get ctx: ', ctx)
        rsp1 = await self.agent_fix.ainvoke({
            'messages': ctx["records"], 
            'profile': self.teacher_profile, 
            'response': ctx["messages"][-1].content,
        })
        #print('llm fix answer with suggestion: ', rsp1)
        return {'messages': [('assistant', rsp1)]}

    async def node_out(self, ctx: MyMessagesState):
        rsp = await self.agent_out.ainvoke({
            'messages': ctx["records"], 
            'rag_data': ctx["rag_data"],
            'fix': ctx["messages"][-1].content, 
            'response': ctx["first_rsp"],
        })
        print('llm final answer: ', rsp)
        ctx["messages"] = ctx["messages"][:-2]
        return {'messages': [('assistant', rsp)]}

    def init_agents(self):
        llm = self.get_llm()
        
        teacher_profile = self.teacher_profile

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", teacher_profile + '\n学校相关基础设施信息：{rag_data}'),
                ("placeholder", "{messages}"),
            ]
        )
        prompt_end = ChatPromptTemplate.from_messages(
            [
                ("system", teacher_profile + '\n学校相关基础设施信息：{rag_data}'+
                '\n请根据聊天记录和修改意见，尽量改善你准备对用户做出的回答，输出最终决定回答的内容。'+
                "\n聊天记录：{messages}\n 修改意见：{fix}\n 之前准备做出的回答：{response}\n 改善后的回答：" ), 
            ]
        )
        prompt_gate = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个负责招生的老师，请根据以下与用户的聊天记录，判断回答用户最后的对话是否需要查阅关于学校基础设施的材料。如果需要请调用相关工具查询相关信息；"
                "如果不需要，请回答:此次回答用户不涉及学校基础设施资料,可直接回答。 \n聊天记录：{messages}\n"),
            ]
        )
        prompt_fix = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个善于为学校招生并且具有良好的沟通能力的老师，"
                "请根据以下与用户的聊天记录和你的职业特点判断后续的这个回答是否合适，并且提出改善的意见."
                "\n聊天记录：{messages}\n 职业特点：{profile}\n 回答：{response}\n "),
            ]
        )
        self.agent_gate = prompt_gate | llm.bind(temperature=0).bind_tools(TOOLS)
        self.agent_continue = prompt | llm| StrOutputParser()
        self.agent_fix = prompt_fix | llm | StrOutputParser()
        self.agent_out = prompt_end | llm | StrOutputParser()

    def get_workflow(self):
        # Create async workflow
        workflow = StateGraph(MyMessagesState)

        # Add nodes with async functions
        workflow.add_node("node0", self.node_gate)
        workflow.add_node('tool_node', self.node_tool)
        workflow.add_node("node1", self.node_continue)
        workflow.add_node("node2", self.node_fix)
        workflow.add_node("node3", self.node_out)

        workflow.set_entry_point('node0')
        workflow.add_conditional_edges(
            "node0", self.continue_or_end, {"tool_node":'tool_node', "node1":'node1'}
            )
        workflow.add_edge('tool_node', 'node1')
        workflow.add_edge("node1", "node2")
        workflow.add_edge("node2", "node3")
        workflow.add_edge("node3", END)

        llm_workflow = workflow.compile()
        return llm_workflow






# Test code
if __name__ == '__main__':
    async def test_workflow():
        msgs = {'messages': [
            #{'role': 'system', 'content': '你是一位历史学家。'}, 
            HumanMessage(content='介绍一下你们学校的宿舍')
            ]
        }
        rst = await WorkFlow().get_workflow().ainvoke(msgs)
        print('\n')
        print(rst)
    asyncio.run(test_workflow())
