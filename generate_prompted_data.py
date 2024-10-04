import sys
from utils import InputDataset


class PromptedDataGenerator():
    def __init__(self,dataset_name,dataset_file,prompt_file,data_to_match=None):
        self.dataset_name = dataset_name
        self.data_to_modify = []
        self.data_to_match = data_to_match
        self.formatted_datasets = ParaphraseDatasetReader().return_data(self.dataset_name,self.dataset_file)
        for d in self.formatted_datasets:
            self.data_to_modify.extend(d)
        with open(prompt_file) as f:
            self.prompt_dict = json.load(f)
    
    def generate_few_shot_data(self,data_folder,special="para"):
        input_data = InputDataset(self.data_to_match).data
        example = ParaphraseDatasetReader().return_data(self.dataset_name)[0]
        if self.dataset_name in ["mrpc","qqp"]:
            ex1 = random.sample(example,1)
            
            pos = [x for x in example if x["label"]=="positive"]
            neg = [x for x in example if x["label"]=="negative"]
            ex2 = (random.sample(pos,1),random.sample(neg,1))
            ex3 = random.sample(example,2)
            ex4 = (random.sample(pos,2),random.sample(neg,2))
        
        else:
            ex1 = random.sample(example,1)
            ex2 = []
            ex3 = random.sample(example,2)
            ex4 = []
        example_dict = {"1":ex1,"2":ex2,"3":ex3,"4":ex4}
        for p in self.prompt_dict.keys():
            new_file_name = self.dataset_name+p+".json"
            outfile = os.path.join(data_folder,new_file_name)
            self.save_new_fewshot(outfile,p,input_data,special=special,example_dict=example_dict)
        return
    
    def generate_new_data(self,data_folder,names=None,downsample=False,special="para"):#change naming to what we need
        if not downsample:
            downsampled_data = InputDataset(self.data_to_match).data
            input_data = downsampled_data
        
        else:
            labels = set([x["label"] for x in self.data_to_modify])
            samples_per_group = int(1000/len(labels))
            selected = []
            for l in labels:
                label_group = [x for x in self.data_to_modify if x["label"]==l]
                group_selected = random.sample(label_group,samples_per_group)
                selected.extend(group_selected)
            random.shuffle(selected)
            input_data = selected
        if downsample:
            down = "_sampled"
        else:
            down = ""
        for p in self.prompt_dict.keys():
            if names==None:
                new_file_name = self.dataset_name+"_"+p+down+"_only_prompt.json"
            else:
                new_file_name = self.dataset_name+"_"+str(names[p])+down+"_only_prompt.json"
            outfile = os.path.join(data_folder,new_file_name)
            self.save_new_prompt_file(outfile,self.prompt_dict[p],input_data,downsample=downsample,special=special)
        return
    
    def save_new_fewshot(self,outfile,p,input_data,special="para",example_dict="none"):
        output_json = []
        for sample in input_data:
            completed_prompt = self.fill_few_shot_prompt(p,sample,special=special,example_dict=example_dict)
            if special=="para":
                output_json_sample = {"prompt":completed_prompt,"sent1":sample["sent1"],"sent2":sample["sent2"],"label":sample["label"]}
            elif special=="common":
                output_json_sample = {"prompt":completed_prompt,"question":sample["question"],"choices":sample["choices"],"label":sample["label"]}
            else:
                pass
            output_json.append(output_json_sample)
        with open(outfile,"w") as f:
            for s in output_json:
                json.dump(s,f,ensure_ascii=False)
                f.write("\n")
        return
    
    def save_new_prompt_file(self,outfile,prompt_text,input_data,downsample=False,special="para"):
        output_json = []
        for sample in input_data:
            completed_prompt = self.fill_prompt(prompt_text,sample,special=special)
            if special=="para":
                output_json_sample = {"prompt":completed_prompt,"sent1":sample["sent1"],"sent2":sample["sent2"],"label":sample["label"]}
            elif special=="common":
                output_json_sample = {"prompt":completed_prompt,"question":sample["question"],"choices":sample["choices"],"label":sample["label"]}
            else:
                pass
            output_json.append(output_json_sample)
        with open(outfile,"w") as f:
            for s in output_json:
                json.dump(s,f,ensure_ascii=False)
                f.write("\n")
        return
    
class PromptSampler():
    def __init__(self,prompted_dataset):
        self.prompted_dataset = prompted_dataset
        self.data = InputDataset(self.prompted_dataset).data
    
    def downsample(self,to_sample=1000,random_sampling=True,class_balance=True):
        if random_sampling:
            if not class_balance:
                selected = random.sample(self.data,to_sample)
            else:
                labels = set([x["label"] for x in self.data])
                samples_per_group = int(to_sample/len(labels))
                selected = []
                for l in labels:
                    label_group = [x for x in self.data if x["label"]==l]
                    group_selected = random.sample(label_group,samples_per_group)
                    selected.extend(group_selected)
                random.shuffle(selected)
        else:
            selected = self.data[:to_sample]
        file_path,file_name = os.path.split(self.prompted_dataset)
        new_file_name = file_name.split(".")[0]+"_sampled.json"
        new_full_path = os.path.join(file_path,new_file_name)
        with open(new_full_path,"w") as f:
            for s in selected:
                json.dump(s,f,ensure_ascii=False)
                f.write("\n")
        return
    
    def match_downsampled(self,downsampled_dataset):
        downsampled_data = InputDataset(downsampled_dataset).data
        sentence_pairs = [(x["sent1"],x["sent2"]) for x in downsampled_data]
        new_downsampled_data = [x for x in self.data if (x["sent1"],x["sent2"]) in sentence_pairs]
        file_path,file_name = os.path.split(self.prompted_dataset)
        new_file_name = file_name.split(".")[0]+"_sampled.json"
        new_full_path = os.path.join(file_path,new_file_name)
        assert len(downsampled_data)==len(new_downsampled_data)
        with open(new_full_path,"w") as f:
            for s in selected:
                json.dump(s,f,ensure_ascii=False)
                f.write("\n")
                
if __name__ == "__main__":
    try:
        dataset_type = sys.argv[1]
        prompt_file = sys.argv[2]
        output_folder = sys.argv[3]
        few_shot = sys.argv[4]
    except:
        raise ValueError("Wrong commandline arguments")  

    if few_shot=="few_shot":
        PD = PromptedDataGenerator(dataset_type,dataset_file,prompt_file).generate_new_data("./DATA/PARAPHRASE/PROMPTED/ROCLING/EXP_1a/test")
    else:
        PD = PromptedDataGenerator(dataset_type,dataset_file,prompt_file).generate_few_shot_data("./DATA/PARAPHRASE/PROMPTED/ROCLING/EXP_3/7")
    