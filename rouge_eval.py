from rouge import Rouge 
rouge_critic = Rouge()

test_file = 'evaluation/topics/predictions/mpt-30b-chat/10_response.txt'
test_file = 'evaluation/topics/predictions/vicuna_7b_flash_seq_4096_to_8192/6_response.txt'
with open(test_file, 'r') as json_file:
    json_list = list(json_file)

for i in range(len(json_list)):
    label = json_list[i].split(',')[0].replace('Label: ', '')
    predict = json_list[i].split(',')[1].replace('Predict: ', '')[3:-2]
    scores = rouge_critic.get_scores(predict, label)[0]
    print(f'Label: {label}')
    print(f'predict: {predict}')
    r = []
    for key in scores.keys():
        for m in ['r', 'p', 'f']:
            r.append(scores[key][m])
    large_scores = [x for x in r if x > 0.5]
    if len(large_scores) >= 3:
        print('Rouge prediction: Correct!')
    elif len(large_scores) <= 1:
        print('Rouge prediction: Wrong!')
    else:
        print('Only two score > 0.5, not sure..')

