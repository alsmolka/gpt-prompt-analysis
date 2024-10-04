Code accompanying our paper on finding salient structural and contextual features in GPT prompt design. You can find the full paper here : []

To generate your own prompted data using our prompt formulas, you can use augment_data.bsh.  If you would like to use the same datasets as in the paper, you need to first download it on you own to ./DATA/TARGET_DATASET folder. You can also use any other dataset without any additional changes as long as it is formatted in the same manner as this benchmark datasets. Simply place the dataset in the data folder and remember to specify it in augment_data.bsh. The default task is paraphrase judgement, but you can change it to multiple choice answering or natural language inference by using a different ready-made prompt file in ./DATA/PROMPTS.

Once you have prompted data, you can run run_GPT.bsh to get GPT answers for the prompts. However, you need to first add you OpenAI API key at the top of GPT_answers.py file. Then run run_GPT.bsh, remembering to specify the folder with prompted samples and the output folder where the GPT answers will be stored.

To evaluate the results, run eval_GPT.bsh. If you simply specify the two files with GPT answers generated using two different formulas, it will print the accuracy for both prompts and additionally compare the performance using the same statistical methods we used in the paper.
 