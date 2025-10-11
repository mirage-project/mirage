import json
import graphviz as gv
import sys
import os
import matplotlib.colors as mcolors
from typing import Literal

base_event = 4294967294

# Exclude these becaues it makes the text hard to see
dark = {"blue", "blueviolet", "brown", "black", "midnightblue", "navy", "indigo", "mediumblue", "dimgrey", "dimgray"}
svg_colors = [color for color in mcolors.CSS4_COLORS.keys() if color not in dark and "dark" not in color]

# Parses the runtime_header.h to get the current task and event types
def get_color_map(prefix: Literal["TASK", "EVENT"]) -> dict[int, tuple[str, str]]:
    # Use MIRAGE_HOME
    filename = os.path.join(os.environ["MIRAGE_HOME"], "include/mirage/persistent_kernel/runtime_header.h")
    with open(filename, "r") as f:
        lines = f.readlines()
    result = {}
    keep_processing = False
    for line in lines:
        if keep_processing and prefix in line:
            name = line.strip().split(" ")[0][len(prefix) + 1:]
            number = int(line.strip().split(" ")[2][:-1])
            result[number] = (svg_colors[len(result)], name)
        elif f"enum {prefix}".upper() in line.upper():
            keep_processing = True
        elif "}" in line:
            keep_processing = False
    print(result)
    return result

# Supported data types and their sizes (in bytes)
supported_data_types = {
    940: ("float16", 2),
    941: ("bfloat16", 2),
    965: ("int64", 8),
}

def check_supported_data_types() -> None:
    filename = os.path.join(os.environ["MIRAGE_HOME"], "include/mirage/type.h")
    with open(filename, "r") as f:
        lines = f.readlines()
    lines_set = set([line.strip() for line in lines])
    for (number, (name, _)) in supported_data_types.items():
        assert f"DT_{name.upper()} = {number}," in lines_set, f"DT_{name.upper()} = {number} is not found in {filename}"
        
# Offset is in bytes, so divide by the size of each element
def get_offset_in_number_of_elements(tensor_json: dict) -> int:
    assert tensor_json["data_type"] in supported_data_types
    return tensor_json["offset"] / supported_data_types[tensor_json["data_type"]][1]


def get_index_from_id(id: int) -> int:
    return id & 0xffffffff

def display_task_graph(task_graph_json_filename: str, use_xdot: bool) -> None:
    check_supported_data_types()
    task_type_color_map = get_color_map("TASK")
    event_type_color_map = get_color_map("EVENT")
    with open(task_graph_json_filename, 'r') as file:
        task_graph = json.load(file)
        g = gv.Digraph()
        g.attr(rankdir="LR")
        num_events = len(task_graph["all_events"])
        num_tasks = len(task_graph["all_tasks"])
        missing_events = set()
        for event_idx, event in enumerate(task_graph["all_events"]):
            description = f"event_idx: {event_idx}\nevent_type: {event_type_color_map[event['event_type']][1]}\nnum_triggers: {event['num_triggers']}\nfirst_task_id: {event['first_task_id']}\nlast_task_id: {event['last_task_id']}"
            g.attr("node", fillcolor=event_type_color_map[event['event_type']][0], style="filled")
            g.node(f"event_{event_idx}", description)
        g.node(f"event_{base_event}", "Base Event")
        g.attr("node", shape="rectangle")
        for task_idx, task in enumerate(task_graph["all_tasks"]):
            inputs_len = str(len(task['inputs'])) if task['inputs'] is not None else "None"
            outputs_len = str(len(task['outputs'])) if task['outputs'] is not None else "None"
            if task["inputs"] is not None:
                input_str = "\n".join([f"in: {input['base_ptr']} + {get_offset_in_number_of_elements(input)}" for input in task["inputs"]])
            else:
                input_str = f"in: None"
            if task["outputs"] is not None:
                output_str = "\n".join([f"out: {output['base_ptr']} + {get_offset_in_number_of_elements(output)}" for output in task["outputs"]])
            else:
                output_str = f"out: None"
            description = f"task_idx: {task_idx}\n{input_str}\n{output_str}\ntask_type: {task_type_color_map[task['task_type']][1]}\nvariant_id: {task['variant_id']}"
            g.attr("node", fillcolor=task_type_color_map[task['task_type']][0], style="filled")
            g.node(f"task_{task_idx}", description)

            dependent_event_idx = get_index_from_id(task['dependent_event'])
            trigger_event_idx = get_index_from_id(task['trigger_event'])
            g.edge(f"event_{dependent_event_idx}", f"task_{task_idx}")
            g.edge(f"task_{task_idx}", f"event_{trigger_event_idx}")

        dot_filename = task_graph_json_filename.replace(".json", ".dot")
        g.save(dot_filename)
        print(f"Graph's dot representation saved as {dot_filename}")
        # Open with xdot
        if use_xdot:
            os.system(f"xdot {dot_filename}")
        else:
            print(f"Consider installing a dot file viewer such as xdot to view/search the graph")
        print("Done rendering")

if __name__ == "__main__":
    # take the argument from the command line
    task_graph_json_filename = sys.argv[1]
    display_task_graph(task_graph_json_filename, use_xdot=True)