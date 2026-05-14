import sys
import os

from sandbox.log import Log, PlanningLog, ConclusionLog

from sandbox.agent import (
    Agent,
    HijackAgent,
    PlanningAgent,
    ConclusionAgent,
    DefenceAgent,
)
from sandbox.pre_info import (
    AgentInfo,
    PlanningAgentInfo,
    ConclusionAgentInfo,
)
from sandbox.edge_finder import (
    find_init_topk_edges,
    update_topk_edges,
)
import random
import json
from tqdm import tqdm
import asyncio

from sandbox.utils import (
    eval_step_misinfo,
    eval_complete_misinfo,
    eval_final_misinfo,
    eval_task_success
)

from sandbox.llm import MODEL_LIST


def read_jsonl(file_path):
    clean_data = []
    with open(file_path , 'r') as f:
        for line in f:
            clean_data.append(json.loads(line))
    return clean_data


class Simulator:

    def __init__(
        self,
        args,
        instance_id: int = 0,
        agents_mode: str = 'preset',
        # model: str = 'gpt-4o-mini',
        planner_model: str = 'gpt-4o-mini',
        executor_model: str = 'gpt-4o-mini',
        topo: str = 'auto',
        attack_method: str = None,
        defense: bool = False,
        syco: bool = False,
        additional_number: int = 1,
        rewrite_argument: bool = False,
    ) -> None:
        self.args = args
        self.id = instance_id
        self.agents_mode = agents_mode
        self.topo = topo
        self.history = []
        self.agents = []
        self.plan_agent = None
        self.conclu_agent = None
        self.tools = []
        self.defense = defense
        self.attack_method = attack_method
        self.use_rag = True if self.attack_method == "rag" else False
        self.syco = syco
        self.additional_number = additional_number
        self.rewrite_argument = rewrite_argument
        self.parttern = args.parttern
        self.dataset = args.dataset
        self.data = self.get_dataset()[instance_id] 

        self.hijack_index = self.args.hijack_index

        self.num_agents = -1
        
        self.planner_model = MODEL_LIST[planner_model]
        self.executor_model = MODEL_LIST[executor_model]


        root_path = f"logs/{self.dataset}/log_hijack_P_{self.planner_model['nickname']}_E_{self.executor_model['nickname']}_step_{self.args.time_step}/{self.parttern}"

        if not os.path.exists(root_path):
                os.makedirs(root_path)
        
        self.log_path = root_path + f"/{instance_id}_hijack_{self.hijack_index}_P_{self.planner_model['nickname']}_E_{self.executor_model['nickname']}_{self.attack_method}_False_mis_ori.jsonl"                
        print(f"Log path set to: {self.log_path}")
       
        self.scores = {"id": self.id, "step_score": [], "complete_score": [], "final_score": [], "task_success": []}
        

    def _init_tools(
        self,
    ) -> None:
        for tool in self.data['tools']:
            self.tools.append(tool)


    def _init_agents(
        self,
    ):
        plan_info = PlanningAgentInfo(1, 0.8, None, self.data, self.topo=='auto',self.args.number_mode)
       
        self.plan_agent = PlanningAgent(
            id=self.id,
            name='Planner',
            model=self.planner_model,
            tools=[],
            background=plan_info,
            simulator=self,
            use_rag=False,
            log_path=self.log_path,
        )

        v_path = f"./logs/{self.dataset}/log_vanilla_P_{self.planner_model['nickname']}_E_{self.executor_model['nickname']}_step_{self.args.time_step}/vanilla/{self.id}_{self.planner_model['nickname']}_{self.executor_model['nickname']}_vanilla_False.jsonl"

        task_view = read_jsonl(v_path)[0]['context']
        plan = read_jsonl(v_path)[1]['context']
        topo = read_jsonl(v_path)[3]['context']
        
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps({"id": self.id, "role": "Planning Agent", "context": task_view}) + '\n')
            file.write(json.dumps({"id": self.id, "role": "Planning Agent", "context": plan}) + '\n')
        

        print(f"Generated Plan: {plan}")
        print(f"Generated Topology: {topo}")

        self.num_agents = len(topo)
       
        random.seed(self.id)
        print(f"Selected hijack agent index: {self.hijack_index}")
        hijack_index = self.hijack_index
        
        
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps({"id": self.id, "infect": None if self.attack_method == "vanilla" or self.attack_method == "rag" else hijack_index, "attack_method": self.attack_method}) + '\n')
       
        for i in range(0, self.num_agents):
            
            if self.attack_method == 'inject':
                if i == hijack_index:
                    plan[i]['user_input'] = self.data['user_input'] 
                    info = AgentInfo(
                        1, 
                        0.3,
                        data=plan[i]
                    )
                    agent = HijackAgent(
                        id=self.id,
                        name=i,
                        model=self.executor_model,
                        tools=self.tools,
                        background=info,
                        simulator=self,
                        use_rag=self.use_rag,
                        extra_data=self.data,
                        log_path=self.log_path,
                    )
                    self.agents.append(agent)
                 
                    continue

            plan[i]['user_input'] = self.data['user_input']

            info = AgentInfo(
                1,
                0.3,
                data=plan[i]
            )
            
            agent = Agent(
                id=self.id,
                name=i,
                model=self.executor_model,
                tools=self.tools,
                background=info,
                simulator=self,
                use_rag=self.use_rag,
                extra_data=self.data,
                poison_rag=True if self.attack_method == "rag" else False,
                log_path=self.log_path,
            )
            self.agents.append(agent)
        
        topo = self._init_neighbors(self.num_agents, topo)
        return topo


    def _init_conclu_agent(
        self,
        complete_log
    ):
        conclu_info = ConclusionAgentInfo(1, 0.8, None, self.data, complete_log)
        self.conclu_agent = ConclusionAgent(
            id=self.id,
            name='Concluder',
            model=self.executor_model,
            tools=[],
            background=conclu_info,
            simulator=self,
            use_rag=False,
            log_path=self.log_path,
        )


    def _init_neighbors(
        self,
        num_agents: int,
        topo = None
    ):

        if self.topo == 'auto':
            if topo is not None:
                if isinstance(topo, list):
                    topo_map = {agent_info["id"]: agent_info["neighborhood"] for agent_info in topo}
                    for i in range(num_agents):
                        if i in topo_map:
                            neighbors = topo_map[i]
                            if neighbors != []:
                                for j in neighbors:
                                    if 0 <= j < num_agents:
                                        self.agents[i].background.neighbors.append(j)
                else:
                    print("Error: Invalid topology format. Expected a list of dictionaries.")
            
            return topo
        elif self.topo == 'chain':
            topo_map = []
            for i in range(num_agents):
                if i == 0:
                    self.agents[i].background.neighbors.append(i+1)
                    topo_map.append({"id": i, "neighborhood": [i+1]})
                    continue
                if i == num_agents - 1:
                    self.agents[i].background.neighbors.append(i-1)
                    topo_map.append({"id": i, "neighborhood": [i-1]})
                    continue
                self.agents[i].background.neighbors.append(i-1)
                self.agents[i].background.neighbors.append(i+1)
                topo_map.append({"id": i, "neighborhood": [i-1, i+1]})
            return topo_map
        elif self.topo == 'full':
            topo_map = []
            for i in range(num_agents):
                topo_map.append({"id": i, "neighborhood": []})
                for j in range(num_agents):
                    if i == j:
                        continue
                    self.agents[i].background.neighbors.append(j)
                    topo_map[i]["neighborhood"].append(j)
            return topo_map
        else:
            raise ValueError(f"Unknown topology type: {self.topo}. Supported types are 'auto', 'chain', and 'full'.")


    def initialize(
        self,
    ):
        
        self._init_tools()
        agent_list = self._init_agents()
        topo_log = PlanningLog(self.id, context=agent_list)
        with open(self.log_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(topo_log.convert_to_json()) + '\n')
        return agent_list

    def get_entrance(
        self,
    ) -> Agent:
        entrance_num = random.randint(0, len(self.agents))
        return self.agents[entrance_num]


    def get_dataset(
        self,
    ):

        data_path = f"dataset/{self.dataset}_Steering.json"
        print(f"Loading dataset from: {data_path}")
        with open(data_path, 'r') as file:
            dataset = json.load(file)
        print(f"Loaded dataset from {data_path}")
        return dataset


    async def emulate(
        self,
        num_step,
        agent_list,
        print_prompt = False,
        print_log = False,
    ) -> None:
        
        complete_log = []
        message_log = []
        
        
        for step in tqdm(range(num_step), desc=f"Running Instance {self.id}", position=1, leave=False):
            with open(self.log_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps({"id": self.id, "step_num": step+1}) + '\n')
            temp_log = []
            
            
            asyncio_tasks = []
            
            for agent in self.agents:
                coro = agent.emulate_one_step(print_prompt, print_log)
                
                asyncio_tasks.append(coro)
            asynio_results = await asyncio.gather(*asyncio_tasks, return_exceptions=True)

            print("Single Round Finished!")
            for i, result in enumerate(asynio_results):
                agent_name = self.agents[i].name
                if isinstance(result, Exception):
                    print(f"Error in Agent {agent_name} during step {step}: {result}")
                    temp_log += [] 
                    message_log += []
                else:
                    once_log, once_message_log = result
                    temp_log += once_log
                    message_log += once_message_log
           
            complete_log += temp_log
        self._init_conclu_agent(complete_log)
        final_answer = await self.conclu_agent.emulate_one_step(print_prompt, print_log)
        