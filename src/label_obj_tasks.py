from collections import defaultdict
import json

obj_tasks = defaultdict(set)
with open("data/taskgrasp/task1_results.txt") as f:
    for line in f.read().strip().splitlines():
        obj, task, classification = line.split("-")
        obj = obj.split("_", 1)[1]
        if classification == "True":
            obj_tasks[obj].add(task)
n_labels = sum(1 for tasks in obj_tasks.values() if len(tasks) > 1)
print(f"Total objects classes: {len(obj_tasks)}, needs labeling: {n_labels}")

ret = {}
for i, (obj, tasks) in enumerate(obj_tasks.items()):
    tasks = sorted(tasks)
    result = None if len(tasks) > 1 else tasks[0]
    while result not in tasks:
        print(f"\n[{i+1: 2d}/{len(obj_tasks)}] Object: '{obj}', Verbs: {tasks}")
        result = input("What is the most appropriate thing to do with this object? ")
    ret[obj] = result

with open("data/taskgrasp/task1_results.json", "w") as f:
    json.dump(ret, f, indent=2)
