import json
import os
import argparse
from tqdm import tqdm
import sys
from SI import get_SI_results
# from gen_utils import *

def find_current_sub_task(original_subtasks, subjective):
    for item in original_subtasks:
        if item['agent_id'] == subjective:
            subtask_name = item['subtask_name']
            subtask_description = item['subtask_description']
            return f"{subtask_name}"
    return "Subtask description not found."

def read_jsonl(file_path, attribute_name):
    clean_data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return [], {}, {}
        
    with open(file_path , 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            clean_data.append(data)
            
    number_list = []
    current_number = 0
    with open(file_path , 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if "step_num" in data:
                number_list.append(current_number)
            current_number += 1
        number_list.append(len(clean_data)-1)
        
    round_data = {}
    
    if len(number_list) > 1:
        for index in range(len(number_list)-1):
            start = number_list[index]+1
            end = number_list[index+1]
            round_data[f"round_{index}"] = clean_data[start: end]
            
    return clean_data, clean_data[3] if len(clean_data) > 3 else {}, round_data

def find_valid_edge(edge_eff_all, ascending=True, exclude_edge=None):
    sorted_edges = edge_eff_all.sort_values('delta_mean', ascending=ascending).reset_index(drop=True)
    for _, row in sorted_edges.iterrows():
        edge = row['edge_eff']
        input_id, output_id = map(int, edge.split('->'))
        if input_id == output_id or input_id == -1 or output_id == -1:
            continue
        if exclude_edge and edge == exclude_edge:
            continue
        return edge
    return None

def find_edge(edge_eff_all, args, type = "max", edge = ""):

    min_row = edge_eff_all.loc[edge_eff_all['delta_mean'].idxmin()]
    min_edge_eff = min_row['edge_eff']

    max_row = edge_eff_all.loc[edge_eff_all['delta_mean'].idxmax()]
    max_edge_eff = max_row['edge_eff']
    
    
    if type == "max":
        max_edge_eff = find_valid_edge(edge_eff_all, ascending=False)
        min_edge_eff = find_valid_edge(edge_eff_all, ascending=True)
    elif type == "min":
        min_edge_eff = find_valid_edge(edge_eff_all, ascending=True, exclude_edge=edge)
        max_edge_eff = find_valid_edge(edge_eff_all, ascending=False, exclude_edge=edge)
    
    strength_input_task_id = int(max_edge_eff.split("->")[0])
    strength_output_task_id = int(max_edge_eff.split("->")[1])
    weak_input_task_id = int(min_edge_eff.split("->")[0])
    weak_output_task_id = int(min_edge_eff.split("->")[1])

    return strength_input_task_id, strength_output_task_id, weak_input_task_id, weak_output_task_id


def main(args):
    start_id = args.start_id   
    end_id = args.end_id    
    save_path = f"../dataset/MisinfoTask_Steering.json"
    print(f"Checking if file exists: {save_path}")
    if not os.path.exists(save_path):
        print(f"Error: {save_path}")
        return
    print(f"Loading data from {save_path}...")
    with open(save_path, "r", encoding='utf-8') as f:
        all_results = json.load(f)
    selected_ids = []
    target_range = [str(i) for i in range(start_id, end_id + 1)] 

    for item in all_results:
        item_id = str(item['id'])
        if item_id in target_range:
            selected_ids.append(item_id)

    if not selected_ids:
        return

    for instance_id in tqdm(selected_ids, desc="Running Instances"):
        print(f"\n=== Processing Instance ID: {instance_id} ===")
        
        current_data = None
        current_data_index = -1
        for idx, item in enumerate(all_results):
            if str(item['id']) == str(instance_id):
                current_data = item
                current_data_index = idx
                break
        
        if current_data is None:
            continue

       
        vanilla_log = f"../logs/{args.dataset}/log_vanilla_P_{args.planner_model}_E_{args.executor_model}_step_3/vanilla/{instance_id}_{args.planner_model}_{args.executor_model}_vanilla_False.jsonl"
        original_results, original_graph_structure, original_round_data = read_jsonl(vanilla_log, "r")
        
        if not original_results:
            print(f"Skipping {instance_id} due to missing vanilla log.")
            continue
        
        hijack_res_path = f"../Diagnosis_results/output_level_score/{instance_id}_{args.planner_model}_{args.executor_model}_inject_False.json"
        
        if not os.path.exists(hijack_res_path):
             print(f"Skipping {instance_id} due to missing hijack eval.")
             continue
       
        single_agent_results = json.load(open(hijack_res_path, "r", encoding='utf-8'))
        
        original_subtasks = original_results[1]['context']

        most_vulnerable_agent = None

        best_score = -1
        agent_id = -1

        per_agent_score = []
        for key in single_agent_results:
            if key == "vanilla": continue
            reference_score = single_agent_results[key]['reference_score']
            misinfo_score = single_agent_results[key]['misinfo_score']
            score_diff = misinfo_score + (10 - reference_score)
            per_agent_score.append((key, score_diff))

        highest_SI_agent_id = max(per_agent_score, key=lambda x: x[1])[0]
        highest_SI_agent_id = int(highest_SI_agent_id.split("_")[1])
        highest_SI_subtask = find_current_sub_task(original_subtasks, highest_SI_agent_id)
        print(f"Highest SI agent: {highest_SI_agent_id} | subtask: {highest_SI_subtask}")
        all_results[int(instance_id)]['Highest_SI_Subtask'] = highest_SI_subtask
        
        
        Lowest_SI_agent_id = min(per_agent_score, key=lambda x: x[1])[0]
        Lowest_SI_agent_id = int(Lowest_SI_agent_id.split("_")[1])
        Lowest_SI_subtask = find_current_sub_task(original_subtasks, Lowest_SI_agent_id)
        print(f"Lowest SI agent: {Lowest_SI_agent_id} | subtask: {Lowest_SI_subtask}")

        dge_eff_all = get_SI_results(instance_id, args)

        max_dge_eff_all = dge_eff_all[dge_eff_all['scenario'] == "hijack_{}".format(highest_SI_agent_id)]
        min_dge_eff_all = dge_eff_all[dge_eff_all['scenario'] == "hijack_{}".format(Lowest_SI_agent_id)]

        strength_input_task_id, strength_output_task_id, _, _ = find_edge(max_dge_eff_all, args, "max")
        _, _, weak_input_task_id, weak_output_task_id = find_edge(min_dge_eff_all, args, "min", edge="{}->{}".format(strength_input_task_id, strength_output_task_id))
        # print(f"Most vulnerable agent's strongest edge: {strength_input_task_id} -> {strength_output_task_id}")
        # print(f"Most robust agent's weakest edge: {weak_input_task_id} -> {weak_output_task_id}")
        strength_input_task = find_current_sub_task(original_subtasks, strength_input_task_id)
        strength_output_task = find_current_sub_task(original_subtasks, strength_output_task_id)
        weak_input_task = find_current_sub_task(original_subtasks, weak_input_task_id)
        weak_output_task = find_current_sub_task(original_subtasks, weak_output_task_id)

        all_results[int(instance_id)]['Preserve_In_task'] = strength_input_task
        all_results[int(instance_id)]['Preserve_Out_task'] = strength_output_task
        all_results[int(instance_id)]['Reduce_In_task'] = weak_input_task
        all_results[int(instance_id)]['Reduce_Out_task'] = weak_output_task

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targeted Subtask Misinfo Generation")
    parser.add_argument("--instance_id", type=int, default=1, help="ID of instance (if single run needed, else loops list)")
    parser.add_argument("--target", type=str, default='low')
    parser.add_argument("--start_id", type=int, default=0, help="Start ID for batch processing")
    parser.add_argument("--end_id", type=int, default=0, help="End ID for batch processing")
    parser.add_argument("--dataset", type=str, default="MisinfoTask", help="Dataset name")
    parser.add_argument("--planner_model", default="gpt-4o-mini", help="LLM")
    parser.add_argument("--executor_model", default="gpt-4o-mini", help="LLM")
    args = parser.parse_args()
    main(args)