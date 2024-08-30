import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10), dpi=300)
para = ['0.01', '0.05', '0.1', '0.2', '0.5', '1', '5', '10']
chebynet = [0.8624, 0.8962, 0.9136, 0.8972, 0.8920, 0.5322, 0.5000, 0.5000] #ChebyNet
gat = [0.8814, 0.8483, 0.8835, 0.8718, 0.8500, 0.5000, 0.5000, 0.5000] #GAT
gcn = [0.7843, 0.8166, 0.8314, 0.8072, 0.8154, 0.5965, 0.5000, 0.5000] #gcn
graphsage = [0.8777, 0.8371, 0.8892, 0.8232, 0.8558, 0.4732, 0.5000, 0.5000] #graphsage

plt.plot(para, chebynet, c='#FFD700', label="ChebyNet")#golden
plt.plot(para, gat, c='#FF7F50', linestyle=':', label="GAT")#coral
plt.plot(para, gcn, c='#32CD32', linestyle='-.', label="GCN")#limegreen
plt.plot(para, graphsage, c='#00BFFF', linestyle='--', label="GraphSAGE")#deepskyblue

plt.scatter(para, chebynet, c='#DAA520')
plt.scatter(para, gat, c='#CD5C5C')
plt.scatter(para, gcn, c='#2E8B57')
plt.scatter(para, graphsage, c='#4682B4')

plt.legend(loc='best',fontsize=18)
#plt.yticks(range(0, 1))
#plt.xlim([0.0, 1.0])
plt.ylim([0.40, 1.05])
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("penalty ($a$)", fontdict={'size': 24})
plt.ylabel("AUC", fontdict={'size': 24})
#plt.title("Hyper-parameter Search", fontdict={'size': 36})
plt.savefig('para7.',dpi=500)
plt.show()
