import argparse
import json
import sys




def read_jsonl(file_path, attribute_name):
    print(f"Reading data from {file_path} for attribute '{attribute_name}'")
    clean_data = []
    
    with open(file_path , 'r') as f:
        for line in f:
            data = json.loads(line)
            clean_data.append(data)
    with open(file_path , 'r') as f:
        current_number = 0
        number_list = []
        for line in f:
            data = json.loads(line)
            if "step_num" in data:
                number_list.append(current_number)
            current_number += 1
        number_list.append(len(clean_data)-1)
    round_data = {}

    for index in range(len(number_list)-1):
        start = number_list[index]+1
        end = number_list[index+1]
        round_data[f"round_{index}"] = clean_data[start: end]
    return clean_data, clean_data[3], round_data

def filter_unique_subjective(data_list):
   
    seen = set()
    result = []
    for item in data_list:
        subj = item.get('subjective')
        obj = item.get('objective')
        if obj is None:
            continue
        if subj not in seen:
            seen.add(subj)
            result.append(item)
    return result
    
def main(args):
    
    for instance_id in range(args.instance_id, args.end_id+1):
        print(f"Instance ID: {instance_id}")

        v_path = f"./logs/{args.dataset}/log_vanilla_P_{args.planner_model}_E_{args.executor_model}_step_3/vanilla/{instance_id}_{args.planner_model}_{args.executor_model}_vanilla_False.jsonl"
        print(v_path)

        _, agent_number, _ = read_jsonl(v_path, "r")

        files = [v_path]
        for hijack_id in range(len(agent_number['context'])):
            files.append(f"./logs/{args.dataset}/log_hijack_P_{args.planner_model}_E_{args.executor_model}_step_3/hijack/{instance_id}_hijack_{hijack_id}_P_{args.planner_model}_E_{args.executor_model}_inject_False_mis_ori.jsonl")


        for file in files:
            print(f"Processing file: {file}")
            
            vanilla_path = file

            vanilla_data, vanilla_structure, vanilla_rounds = read_jsonl(vanilla_path, "r")

            vanilla_filtered_round_data = {}
            for index in range(0,len(vanilla_rounds)):
                vanilla_filtered_round_data[f"round_{index}"] = sorted(filter_unique_subjective(vanilla_rounds[f"round_{index}"]), key=lambda x: x.get('subjective', float('inf')))
            
            vanilla_results = {}
          
            results_path = f"./Diagnosis_results/node_level_score/{file.split('/')[-1].replace('.jsonl', '.json')}"
            print(f"Saving results to {results_path}...")
            # continue
            save_path = results_path

            result_data = json.load(open(results_path, 'r'))            

            for index in range(0,len(vanilla_filtered_round_data)):
                if index == 0:
                    for key in result_data[f"round_{index}"].keys():
                        result_data[f"round_{index}"][key]['prev'] = int(key)
                else:
                    current_round_data = vanilla_filtered_round_data[f"round_{index}"]
                    all_prev_data = []
                    for prev_idx in range(index):
                        all_prev_data.extend(vanilla_filtered_round_data.get(f"round_{prev_idx}", []))
                    sorted_data = sorted(current_round_data, key=lambda x: x.get('subjective', float('inf')))
                    print(f"Subjective values in round {index} (sorted):")

                    for item in sorted_data:
                        subj = item.get('subjective')
                        context = item.get('context')
                        # print(f"  subjective: {subj}, context: {context}")
                        print(f"  subjective: {subj}")
                        
                        recv_ctx = item.get('receive_context')
                        
                        found = None
                        for prev_item in all_prev_data:
                            if prev_item.get('context') == recv_ctx:
                                found = prev_item.get('subjective')
                                break
                        print(f"  receive_context matches previous round subjective: {found}")
                        
                        result_data[f"round_{index}"][str(subj)]['prev'] = found
                        
            with open(save_path, 'w') as f:
                json.dump(result_data, f, indent=4)
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Influence Analysis")
    parser.add_argument("--hijack_agent", type=int, default=0, help="ID of the agent to hijack")
    parser.add_argument("--instance_id", type=int, default=0, help="ID of the instance to analyze")
    parser.add_argument("--end_id", type=int, default=0, help="ID of the instance to analyze")
    parser.add_argument("--dataset", type=str, default="MisinfoTask", help="Dataset name")
    parser.add_argument("--planner_model", default="gpt-4o-mini", help="LLM")
    parser.add_argument("--executor_model", default="gpt-4o-mini", help="LLM")
    args = parser.parse_args()
    main(args)

