import ujson as json

class InputDataset():
    def __init__(self,infile) -> None:
        self.data = []
        with open(infile) as f:
            for l in f.readlines():
                #print(l)
                self.data.append(json.loads(l))
    def get_data(self,len_lim=True):
        if not len_lim:
            result = self.data
        else:
            result = [x for x in self.data if len(x["sent1"].split(" "))<22 and len(x["sent2"].split(" "))<22]
        return result
    

class ParaphraseDatasetReader():
    def __init__(self):
        pass
            
    def return_data(self,dataset_type,*dataset_files):
        train = []
        dev = []
        test = []
        if dataset_type=="hans":
            train,dev,test = self.read_hans([dataset_files])
        if dataset_type=="snli":
            train,dev,test = self.read_snli([dataset_files])
        if dataset_type=="paws":
            train,dev,test = self.read_paws([dataset_files])
        if dataset_type=="webis":
            train,dev,test = self.read_webis([dataset_files])
        if dataset_type=="subtrees":
            train,dev,test = self.read_subtrees([dataset_files])
        if dataset_type=="qqp":
            train,dev,test = self.read_qqp([dataset_files])
        if dataset_type=="ppdb":
            train,dev,test = self.read_ppdb([dataset_files])
        if dataset_type=="mrpc":
            train,dev,test = self.read_mrpc([dataset_files])
        if dataset_type=="mrpc_down":
            train,dev,test = self.read_mrpc_down(dataset_files])
        if dataset_type=="commonsense":
            train,dev,test = self.read_commonsense()
        if dataset_type=="squad":
            train,dev,test = self.read_squad([dataset_files])
        return train, dev,test
    
    def read_commonsense(self):
        train = []
        ds = load_dataset("tau/commonsense_qa",split="train")
        for s in ds:
            c = s["choices"]
            combined_choices = []
            for alt in zip(c["label"],c["text"]):
                combined_choices.append(" ".join([alt[0],alt[1]]))
            combined_choices = " ".join(combined_choices)
            train.append({"question":s["question"],"choices":combined_choices,"label":s["answerKey"]})
        return train,[],[]
    
    def read_squad(self,file_list):
        pass
    
    def read_paws(self,file_list):
        #3 paws files (train,dev,test)
        train = []
        with open(file_list[0]) as f:
            df_train = pd.read_csv(f,sep="\t")
        with open(file_list[1]) as f:
            df_test = pd.read_csv(f,sep="\t")
        df = pd.concat([df_train,df_test], axis=0)
        positive = df[df['label']==1]
        negative = df[df['label']==0]
        for index, row in positive.iterrows():
            train.append({"sent1":row['sentence1'],"sent2":row['sentence2'],"label":"positive"})
        for index, row in negative.iterrows():
            train.append({"sent1":row['sentence1'],"sent2":row['sentence2'],"label":"negative"})
        return train,[],[]
    
    def read_ppdb(self,file_list):

        with open(file_list[0]) as f:
            line_list = f.readlines()
        full_samples = [PPDBSample(l).para for l in line_list]
        #print(full_samples)
        train = [{"sent1":x[0],"sent2":x[1],"label":"positive"} for x in full_samples]
        return train,[],[]
            
    def read_qqp(self,file_list):
        train = []
        with open(file_list[0]) as f:
            df = pd.read_csv(f,sep="\t")
        df.dropna(how='any', inplace=True)
        positive = df[df['is_duplicate']==1]
        negative = df[df['is_duplicate']==0]
        
        for index, row in positive.iterrows():
            train.append({"sent1":row['question1'],"sent2":row['question2'],"label":"positive"})
        for index, row in negative.iterrows():
            train.append({"sent1":row['question1'],"sent2":row['question2'],"label":"negative"})
        return train, [], []
    
    def read_subtrees(self,file_list):
        train = []
        for fol in file_list:
            full_samples = InputDataset(file_list[0]).data
            for s in full_samples:
                if len(s["subtree_pairs"])>0:
                    train.extend({"sent1": " ".join(x[0]),"sent2": " ".join(x[1]),"label":"positive"} for x in s["subtree_pairs"])
        data = train
        return data,[],[]
            
    
    def read_hans(self,file_list):
        train = []
        test = []
        with open(file_list[0]) as f:
            df = pd.read_table(f)
        train_samples,test_samples = train_test_split(df,test_size=0.2,random_state=skl_state)
        #print(train_samples["gold_label"].value_counts())
        #print(test_samples["gold_label"].value_counts())
        for index, row in train_samples.iterrows():
            train.append({"sent1":row["sentence1"],"sent2":row["sentence2"],"label":row["gold_label"]})
        for index, row in test_samples.iterrows():
            test.append({"sent1":row["sentence1"],"sent2":row["sentence2"],"label":row["gold_label"]})
        return train, dev, test
    
        
    def read_snli(self,file_list):
        #make sure only pos and neg are read
        train = []
        test = []
        dev = []
        with open(file_list[0]) as f:
            df_train = pd.read_table(f)
            df_train = df_train[df_train.gold_label!="-"]
            #df_train = df_train.replace("contradiction", "non-entailment") 
        with open(file_list[1]) as f:
            df_test = pd.read_table(f)
            df_test = df_test[df_test.gold_label!="-"]
            #df_dev = df_dev.replace("contradiction", "non-entailment") 

        for index, row in df_train.iterrows():
            train.append({"sent1":row["sentence1"],"sent2":row["sentence2"],"label":row["gold_label"]})
        for index, row in df_test.iterrows():
            test.append({"sent1":row["sentence1"],"sent2":row["sentence2"],"label":row["gold_label"]})
        return train,[],test

    def read_mrpc(self,file_list):
        train = []
        dev = []
        with open(file_list[0]) as f:
            df_train = pd.read_csv(f,sep='\t',on_bad_lines='skip')
        with open(file_list[1]) as f:
            df_dev= pd.read_csv(f,sep='\t',on_bad_lines='skip')
        df_train.dropna(how='any', inplace=True)
        df_dev.dropna(how='any', inplace=True)
        for index, row in df_train.iterrows():
            if row["Quality"]==0:
                label = "negative"
            else:
                label = "positive"
            train.append({"sent1":row["#1 String"],"sent2":row["#2 String"],"label":label})
        for index, row in df_dev.iterrows():
            if row["Quality"]==0:
                label = "negative"
            else:
                label = "positive"
            dev.append({"sent1":row["#1 String"],"sent2":row["#2 String"],"label":label})      
        return train,dev,[]
    
    def read_mrpc_down(self,file_list):
        train = []
        with open(file_list[0]) as f:
            df_train = pd.read_csv(f)
        for index, row in df_train.iterrows():
            if row["label"]==0:
                label = "negative"
            else:
                label = "positive"
            train.append({"sent1":row["sentence1"],"sent2":row["sentence2"],"label":label})
        return train,[],[]

    def read_webis(self,file_list,sentence_file=True):
        file_dict = defaultdict(tuple)
        if sentence_file:
            train = []
            with open(file_list[0]) as f:
                for l in f:
                    train.append(json.loads(l))
        else:
            paragaph_dict = defaultdict(tuple)
            #input - glob glob file list of all webis samples
            org = file_list[0]
            
            par = file_list[1]
            meta = file_list[2]
            org_id = {}
            par_id = {}
            for i in org:
                org_id[i] = self.strip_filename(i)
            for i in par:
                par_id[i] = self.strip_filename(i)
            #filter out positive files (dictionary with file names in tuples - objects? - each object with two paragraphs)
            pos_fileid = [self.strip_filename(x) for x in meta if self.get_pos_id(x)]
            #print(org_id)
            for i in pos_fileid:
                o = [x for x in org if org_id[x]==i][0]
                p = [x for x in par if par_id[x]==i][0]
                file_dict[i] = (o,p)
            #align sentences
            paragraph_dict = {k:(self.read_paragraph(v[0]),self.read_paragraph(v[1])) for k,v in file_dict.items()}
            input_paragraphs = [{"paragraph_1":paragraph_dict[k][0],"paragraph_2":paragraph_dict[k][1]} for k in paragraph_dict.keys()]
            aligned_sentences = SentenceAligner(input_paragraphs).get_aligned_pairs()
            #print("input",input_paragraphs)
            train = [{"sent1":x[0],"sent2":x[1],"label":""} for x in aligned_sentences]
        return train,[],[]
    
    def read_mp(self):
        with open(self.data[0]) as f:
            df_train = pd.read_csv(f,sep="\t",on_bad_lines='skip')
        with open(self.data[1]) as f:
            df_test= pd.read_csv(f,sep="\t",on_bad_lines='skip')
        df = pd.concat([df_train,df_test], axis=0)
        positive = df[df['Quality']==1]
        negative = df[df['Quality']==0]
        pos1,pos2 = positive['#1 String'],positive['#2 String']
        neg1,neg2 = negative['#1 String'],negative['#2 String']
        return list(zip(pos1,pos2)),list(zip(neg1,neg2))

def run_ttest(file1,file2,mode="mrpc"):
    answers1 = Evaluator(file1).run_evaluation(mode)
    answers2 = Evaluator(file2).run_evaluation(mode)
    list_means1 = []
    list_means2 = []
    for i in range(0, 1000, 100):
        small_list1 = answers1[i:i + 100]
        list_means1.append(sum(small_list1)/100)
        small_list2 = answers2[i:i + 100]
        list_means2.append(sum(small_list2)/100)
    #calculate means
    stats = ttest_rel(list_means1,list_means2)
    print(stats)
    #run t-test on two lists of means
    return

if __name__ == "__main__":
    pass