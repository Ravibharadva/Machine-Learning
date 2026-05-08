import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from data_loader import load_heart_disease
from preprocessing import preprocess
from model import train, predict

df, target = load_heart_disease("heart.xls")
X_train, X_test, y_train, y_test, features = preprocess(df, target)

print("Training NN LR...")
model, losses = train(X_train, y_train)
nn_pred, nn_probs = predict(model, X_test)

nn_acc = accuracy_score(y_test, nn_pred)
nn_auc = roc_auc_score(y_test, nn_probs)
print(f"NN LR...")
print(f"Accuracy: {nn_acc:.4f} | ROC-AUC: {nn_auc:.4f}")
print(classification_report(y_test, nn_pred, target_names=["No Disease", "Disease"]))


print(f"Traning SKlearn LR(baseline)..")
sk = LogisticRegression(max_iter=1000, random_state=42)
sk.fit(X_train, y_train)
sk_pred = sk.predict(X_test)
sk_prob = sk.predict_proba(X_test)[:,1]

sk_acc = accuracy_score(y_test, sk_pred)
sk_auc = roc_auc_score(y_test, sk_prob)
print(f"\n sklearn LR...")
print(f"Accuracy: {sk_acc:.4f} | ROC-AUC: {sk_auc:.4f}")
print(classification_report(y_test, sk_pred, target_names=["No Disease", "Disease"]))


#plots
fig, axes = plt.subplots(1,3, figsize=(18, 5))
fig.suptitle("Heart Disease — Logistic Regression via Neural Network",
             fontsize=13, fontweight="bold")

#loss curve
axes[0].plot(losses, color="#363946", linewidth=2)
axes[0].set_title("trainig loss")
axes[0].set_ylabel("BCE LOSS")
axes[0].set_xlabel("epoch")
axes[0].grid(alpha=0.3)

#confusion matrix
cm = confusion_matrix(y_test, nn_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Pred:NO", "pred: Yes"],
            yticklabels=["true:No", "true:yes"])
axes[1].set_title("confusion matrix")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

#ROC Curve
for prob, label, color in [(nn_probs, "NN LR", "#e63946"), (sk_prob, "SK LR", "#457b9d")]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[2].plot(fpr, tpr, label=f"{label}, (AUC={roc_auc_score(y_test, prob):.4f})", color=color, linewidth=2)
axes[2].plot([0,1], [0,1], "k--", linewidth=1)
axes[2].set_title("ROC Curve")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("heart_results.png", dpi=150, bbox_inches="tight")
print("\n Plots saved as heart_results.png")
plt.show()