import sys
import ujson as json

from scipy.stats import ttest_rel
from utils import InputDataset, run_ttest

class Evaluator():
    
    def __init__(self,answers_file,answer_type="paraphrase"):
        self.answers = InputDataset(answers_file).data
        self.answer_type = answer_type
         
    
    def _translate_answer(self,model_answer):
        errors = 0
        dots = 0 
        cut_short = 0
        other = 0
        caps = 0
        trail = 0
        model_answer = model_answer.strip().lower()
        model_answer = model_answer.strip('"')
        model_answer = model_answer.strip('”')
        model_answer = model_answer.strip('“')
        model_answer = model_answer.replace(".","")
        model_answer = "".join([x for x in model_answer if x.isalpha()])
        if model_answer!="paraphrase" and model_answer!="not paraphrase":
            errors+=1
            if model_answer[-1:]=="." and len(model_answer)>1:
                dots+=1
            no_whitespace_model_answer = model_answer.replace(" ","")
            if len(no_whitespace_model_answer) < len(model_answer):
                cut_short+=1
            else:
                other+=1

        if self.answer_type=="paraphrase_split":
                if model_answer=="paraphrase":
                    answer = "positive"
                elif model_answer =="notparaphrase":
                    answer = "negative"
                else:
                    answer = "none"
        if self.answer_type=="paraphrase":
                if model_answer=="paraphrase":
                    answer = "positive"
                elif model_answer =="not-paraphrase":
                    answer = "negative"
                else:
                    answer = "none"
        if self.answer_type=="yes":
                if model_answer=="yes":
                    answer = "positive"
                elif model_answer =="no":
                    answer = "negative"
                else:
                    answer = "none"
        return answer,(errors,dots,cut_short,other)
                
    def run_evaluation(self,mode="mrpc"):
        answers = []
        errors = 0
        dots = 0 
        cut_short = 0
        other = 0
        if mode in ["mrpc","qqp"]:
            correct_pos = 0
            correct_neg = 0
            false_negative = 0
            false_positive  = 0
            all_positive = len([x for x in self.answers if x["label"]=="positive"])
            all_negative = len([x for x in self.answers if x["label"]=="negative"])
            for s in self.answers:
                golden = s["label"]
                model_answer,error_sources = self._translate_answer(s["model_answer"],mode)
                if golden==model_answer and golden=="positive":
                    correct_pos+=1
                    answers.append(1)
                elif golden==model_answer and golden=="negative":
                    answers.append(1)
                    correct_neg+=1
                elif golden!=model_answer and golden=="negative":
                    false_positive+=1
                    answers.append(0)
                elif golden!=model_answer and golden=="positive":
                    false_negative+=1
                    answers.append(0)

            total_acc = (correct_neg+correct_pos)/len(self.answers)
            neg_acc = correct_neg/all_negative
            pos_acc = correct_pos/all_positive
            print("Total accuracy: ", total_acc)
            print("Negative accuracy: ",neg_acc)
            print("Positive accuracy: ",pos_acc)
            print("tp fp tn fn")
            print(correct_pos,false_positive, correct_neg,false_negative)
        
            
        if mode=="snli":
            correct_ent = 0
            correct_cont = 0
            correct_neu = 0
            inc_ent = 0
            inc_cont = 0
            inc_neu = 0
            all_ent = len([x for x in self.answers if x["label"]=="entailment"])
            all_cont = len([x for x in self.answers if x["label"]=="contradiction"])
            all_neu = len([x for x in self.answers if x["label"]=="neutral"])
            for s in self.answers:
                model_answer,error_sources = self._translate_answer(s["model_answer"],mode)
                golden = s["label"]
                if golden=="entailment":
                    if model_answer==golden:
                        correct_ent+=1
                        answers.append(1)
                    else:
                        inc_ent+=1
                        answers.append(0)
                if golden=="contradiction":
                    if model_answer==golden:
                        correct_cont+=1
                        answers.append(1)
                    else:
                        inc_cont+=1
                        answers.append(0)
                if golden=="neutral":
                    if model_answer==golden:
                        answers.append(1)
                        correct_neu+=1
                    else:
                        answers.append(0)
                        inc_neu+=1
            total_acc = (correct_ent+correct_cont+correct_neu)/len(self.answers)
            print(total_acc)
                    
        if mode=="commonsense":
            correct = 0
            for s in self.answers:
                model_answer,error_sources = self._translate_answer(s["model_answer"],mode)
                golden = s["label"].lower()
                if golden==model_answer:
                    correct+=1
                    answers.append(1)
                else:
                    answers.append(0)
            total_acc = correct/len(self.answers)    
            print(total_acc)      
        return answers
    


if __name__ == "__main__":
    try:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        mode = sys.argv[3]
    except:
        raise ValueError("Wrong commandline arguments")  
    run_ttest(file1,file2,mode=mode)