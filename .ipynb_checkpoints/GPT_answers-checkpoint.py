import openai
import ujson as json
import glob
import os

from utils import InputDataset

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  






client = openai.OpenAI(api_key='your_api_key')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.completions.create(**kwargs)

def get_answer(prompt):
    model_answer  = completion_with_backoff(model="gpt-3.5-turbo-instruct",prompt=prompt,max_tokens=10)
    return model_answer.choices[0].text

def run_on_dataset(prompt_file,results_file):
    test_dataset = InputDataset(prompt_file).data
    completed_samples = []
    for i,sample in enumerate(test_dataset):
        print(i)
        model_answer = get_answer(sample["prompt"])
        sample["model_answer"] = model_answer
        completed_samples.append(sample)

    with open(results_file,"w") as f:
        for s in completed_samples:
            json.dump(s,f,ensure_ascii=False)
            f.write("\n")
    return completed_samples


if __name__ == "__main__":
    try:
        prompt_folder = sys.argv[1]
        results_folder = sys.argv[2]
    except:
        raise ValueError("Wrong commandline arguments")        
    prompt_files = glob.glob(prompt_folder+"/*.json")

    for fl in prompt_files:
        prompt_file = fl
        file_path,file_name = os.path.split(fl)
        full_file_name = file_name.split(".")[0]
        new_file_name = full_file_name+".json"
        results_file = os.path.join(results_folder,file_name)
        answers = run_on_dataset(prompt_file,results_file)


    


    

    
   