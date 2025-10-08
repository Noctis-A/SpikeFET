import os

data_path = '/home/work/xxx/VisEvent/test'
save_path = data_path
file_folds = os.listdir(data_path)

for videoID in range(len(file_folds)):
    foldname = file_folds[videoID]
    with open(save_path + '/list.txt', 'a+') as f:
        f.write(foldname+'\n')

# train.txt  generate
# i = 0
# for i in range(0, len(file_folds)):
#     with open(save_path + '/train.txt', 'a+') as f:
#         f.write(str(i)+'\n')
#         i=i+1

# val.txt  generate
i = 0
for i in range(0, len(file_folds)):
    with open(save_path + '/val.txt', 'a+') as f:
        f.write(str(i)+'\n')
        i=i+1