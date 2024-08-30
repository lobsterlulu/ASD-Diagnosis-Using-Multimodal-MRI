import matplotlib.pyplot as plt
import pickle

'GCN'
gcn_file_without_loss = open('./GCN.pkl', 'rb')
data_gcn = pickle.load(gcn_file_without_loss)
fpr, tpr, auc = data_gcn

gcn_file_with_loss = open('./GCN+loss.pkl', 'rb')
data_gcn1 = pickle.load(gcn_file_with_loss)
fpr1, tpr1, auc1 = data_gcn1

'GAT'
gat_file_without_loss = open('./gat.pkl', 'rb')
data_gat = pickle.load(gat_file_without_loss)
fpr2, tpr2, auc2 = data_gat

gat_file_with_loss = open('./gat+loss.pkl', 'rb')
data_gat1 = pickle.load(gat_file_with_loss)
fpr3, tpr3, auc3 = data_gat1

'graphsage'
graphsage_file_without_loss = open('./graphsage.pkl', 'rb')
data_graphsage = pickle.load(graphsage_file_without_loss)
fpr4, tpr4, auc4 = data_graphsage

graphsage_file_with_loss = open('./graphsage+loss.pkl', 'rb')
data_graphsage1 = pickle.load(graphsage_file_with_loss)
fpr5, tpr5, auc5 = data_graphsage1

'chebnet'
chebnet_file_without_loss = open('./chebnet.pkl', 'rb')
data_chebnet = pickle.load(chebnet_file_without_loss)
fpr6, tpr6, auc6 = data_chebnet

chebnet_file_with_loss = open('./ChebNet+loss.pkl', 'rb')
data_chebnet1 = pickle.load(chebnet_file_with_loss)
fpr7, tpr7, auc7 = data_chebnet1

plt.figure()
lw = 2
plt.plot(fpr, tpr, color="#6C8EBF", lw=lw, label="AUC=%0.3f (GCN)" % auc)
plt.plot(fpr1, tpr1, color="#B85450", lw=lw, label="AUC=%0.3f (GCN+$\mathcal{R}_{g}$)" % auc1)
plt.fill_between(fpr1,tpr1,0,color='#F8CECC')
plt.fill_between(fpr,tpr,0,color='#DAE8FC')

plt.plot([0, 1], [0, 1], color="#6F6F6F", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GCN")
plt.legend(loc="lower right")
plt.savefig('GCN_roc.',dpi=500)
plt.show()

plt.figure()
lw = 2
plt.plot(fpr2, tpr2, color="#6C8EBF", lw=lw, label="AUC=%0.3f (GAT)" % auc2)
plt.plot(fpr3, tpr3, color="#B85450", lw=lw, label="AUC=%0.3f (GAT+$\mathcal{R}_{g}$)" % auc3)
plt.fill_between(fpr3,tpr3,0,color='#F8CECC')
plt.fill_between(fpr2,tpr2,0,color='#DAE8FC')

plt.plot([0, 1], [0, 1], color="#6F6F6F", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GAT")
plt.legend(loc="lower right")
plt.savefig('GAT_roc.',dpi=500)
plt.show()

plt.figure()
lw = 2
plt.plot(fpr4, tpr4, color="#6C8EBF", lw=lw, label="AUC=%0.3f (GraphSAGE)" % auc4)
plt.plot(fpr5, tpr5, color="#B85450", lw=lw, label="AUC=%0.3f (GraphSAGE+$\mathcal{R}_{g}$)" % auc5)
plt.fill_between(fpr5,tpr5,0,color='#F8CECC')
plt.fill_between(fpr4,tpr4,0,color='#DAE8FC')

plt.plot([0, 1], [0, 1], color="#6F6F6F", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GraphSAGE")
plt.legend(loc="lower right")
plt.savefig('GraphSAGE_roc.',dpi=500)
plt.show()

plt.figure()
lw = 2
plt.plot(fpr6, tpr6, color="#6C8EBF", lw=lw, label="AUC=%0.3f (ChebyNet)" % auc6)
plt.plot(fpr7, tpr7, color="#B85450", lw=lw, label="AUC=%0.3f (ChebyNet+$\mathcal{R}_{g}$)" % auc7)
plt.fill_between(fpr7,tpr7,0,color='#F8CECC')
plt.fill_between(fpr6,tpr6,0,color='#DAE8FC')

plt.plot([0, 1], [0, 1], color="#6F6F6F", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ChebyNet")
plt.legend(loc="lower right")
plt.savefig('ChebNet_roc.',dpi=500)
plt.show()