import argparse
from sandbox.simulator_hijack import Simulator
from tqdm import tqdm
import asyncio
import sys
import json

def read_jsonl(file_path, attribute_name):
    clean_data = []
    with open(file_path , 'r') as f:
        for line in f:
            clean_data.append(json.loads(line))
    return clean_data

async def run(
    args,
    instance_id,
    planer_model,
    executor_model,
    attack_method,
    topo,
    defense,
    time_step,
    print_prompt,
    print_log
):
    vanilla_datas = read_jsonl(f"./logs/{args.dataset}/log_vanilla_P_{args.planner_model}_E_{args.executor_model}_step_3/vanilla/{instance_id}_{args.planner_model}_{args.executor_model}_vanilla_False.jsonl", "topology")[3]['context']
    
   
    for hijack_id in range(len(vanilla_datas)):
        args.hijack_index = hijack_id
        
        sim = Simulator(args, instance_id, attack_method=attack_method, planner_model=args.planner_model, executor_model=args.executor_model, topo=topo, defense=defense, syco=args.syco, additional_number = args.additional_number, rewrite_argument=args.rewrite_argument)
        
        agent_list = sim.initialize()
        print(agent_list)

        await sim.emulate(time_step, agent_list, print_prompt=print_prompt, print_log=print_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The following are the options used to control the agent system")
    parser.add_argument("--instance_id", type=int, default=1, help="ID of instance")
    parser.add_argument("--defense", default=False, help="Whether to use defense")
    parser.add_argument("--time_step", default=3, type=int, help="Steps to run")
    parser.add_argument("--attack_method", default="vanilla", choices=['vanilla', 'inject', 'tool', 'rag'], help="Method")
    parser.add_argument("--print_prompt", default=False, help="Whether to print prompt")
    parser.add_argument("--print_log", default=False, help="Whether to print log")
    parser.add_argument("--planner_model", default="gpt-4o-mini", help="LLM")
    parser.add_argument("--executor_model", default="gpt-4o-mini", help="LLM")
    parser.add_argument("--topo", default="auto", choices=['auto', 'chain', 'full'], help="The topology of the agent system")
    parser.add_argument("--syco", default=False, action='store_true', help="Whether to use syco dataset")
    parser.add_argument("--additional_number", type=int, default=1, help="The number of additional syco agents")
    parser.add_argument("--rewrite_argument", default=False, action='store_true', help="Whether to rewrite misinfo argument")

    parser.add_argument("--special_number", type=int, default=1, help="number of agents")
    parser.add_argument("--hijack_index", type=int, default=0, help="index of hijack agent")
    parser.add_argument("--end_id", type=int, default=107, help="index of end instance")
    parser.add_argument("--parttern", type=str, default="", help="index of hijack agent")
    parser.add_argument("--target", type=str, default="", help="index of hijack agent")
    parser.add_argument("--number_mode", choices=['auto', 'special', 'default'], default='default', help="Mode for setting number of agents")

    parser.add_argument(
        "--dataset",
        type=str,
        default="MisinfoTask",
        help="dataset name",
        choices=["MisinfoTask", "ASB"],
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")
   
    selected_ids = [i for i in range(0, 110)]

    idx = selected_ids.index(args.instance_id) if args.instance_id in selected_ids else 0
    print(f"Running instances starting from ID: {args.instance_id} (index {idx})")

    for i in tqdm(selected_ids[idx:107], desc="Running Instances", position=0, leave=True):
        print(f"\n=== Running instance {i} ===")
        if i == args.end_id+1: break
        asyncio.run(run(args, i, args.planner_model, args.executor_model, args.attack_method, args.topo, args.defense, args.time_step, args.print_prompt, args.print_log))
