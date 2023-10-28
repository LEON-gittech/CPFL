from sklearn.manifold import TSNE
import numpy as np
import os
import matplotlib.pyplot as plt
# path = "/home/leon/workspace/pFedSD/pcode/global_train/"
# out_file = path+"fedavg_con_out/"
# target_file = path+"fedavg_con_target/"
# # out_file = path+"fedavg_out/"
# # target_file = path+"fedavg_target/"
# tsne = TSNE()

# outs = os.listdir(out_file)
# targets = os.listdir(target_file)

# out_all,target_all = [],[]
# for out, target in zip(outs, targets):
#     out = np.load(out_file+out)
#     target = np.load(target_file+target)
#     out_all.append(out)
#     target_all.append(target)

# out_all = np.concatenate(out_all)
# target_all = np.concatenate(target_all)
# out_all = tsne.fit_transform(out_all)
# np.save('fedavg_con_label.py',target_all)
# np.save('fedavg_con_tsne.py',out_all)
# np.save('fedavg_label.py',target_all)
# np.save('fedavg_tsne.py',out_all)
# out_all = np.load('fedavg_con_tsne.py.npy')[:10]
# target_all = np.load('fedavg_con_label.py.npy')[:10]
out_all = np.load('fedavg_tsne.py.npy')[:10]
target_all = np.load('fedavg_label.py.npy')[:10]
fig = plt.figure()
for i in range(10):
    indices = target_all == i
	# 标签为i的全部选出来

    x, y = out_all[indices].T # 这里转置了

	# 画图
    plt.scatter(x, y, label=str(i))
plt.legend()
# plt.savefig('fedavg_con_tsne.jpg')
plt.savefig('fedavg_tsne.jpg')
plt.show()