import json
import os


f = open("predictions/clip_baseline/results.json")
baseline_accs = json.load(f)
baseline_total_acc = 0.0 
total_acc = 0.0
baseline_total_acc_for_avg = 0.0
total_acc_for_avg = 0.0
num_samples = 0
json_paths = os.path.join("predictions","blip_token_ablation")
num_datasets = 0
for k in baseline_accs:
    filename = k + ".json"
    dataset_path = os.path.join(json_paths,filename)
    if not os.path.isfile(dataset_path):
        continue
    num_datasets += 1
    f = open(dataset_path)
    data = json.load(f)
    prediction_acc = data["accuracy"]
    dataset_size = data["size"]
    baseline_acc = baseline_accs[k]
    baseline_total_acc += baseline_acc * dataset_size
    total_acc += prediction_acc * dataset_size
    baseline_total_acc_for_avg += baseline_acc
    total_acc_for_avg += prediction_acc
    num_samples += dataset_size

out_file = open(os.path.join(json_paths,"summary.json"), "w") 
json.dump({"weighted_prediction_acc": total_acc/num_samples,
 "weighted_baseline_acc": baseline_total_acc/num_samples,
 "average_predicted_acc": total_acc_for_avg/num_datasets,
 "average_baseline_acc": baseline_total_acc_for_avg/num_datasets,
"num_datasets":num_datasets}, out_file, indent = 6)
out_file.close()