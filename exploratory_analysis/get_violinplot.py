import pickle
with open("similarities.pkl", "rb") as f:
    similarities = pickle.load(f)


import matplotlib.pyplot as plt
import numpy as np


labels = [
    "",
    "NPClassifierFP",
    "Biosynfoni",
    "NeuralNPFP",
    "MHFP", 
    "MorganFingerprint",
    "NPBERT"
]

fig, ax = plt.subplots()
ax.set_xticklabels(labels=labels, )
ax.set_xlabel("Fingerprint Method")           
ax.set_ylabel("Similarity Score")   

for i in range(0,similarities.shape[0]):
    ax.violinplot(dataset=similarities[i],positions=[i])
plt.xticks(rotation=45)
plt.title("Distribution of Similarity Scores by Fingerprint")
plt.tight_layout()

plt.savefig("violinplot_similarity.png", dpi=300)



