import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

class EvaluationEngine:
    def __init__(self, model, blockchain, label_encoder):
        self.model = model
        self.blockchain = blockchain
        self.label_encoder = label_encoder
        self.classes = self.label_encoder.classes_
        
    def compute_metrics(self, y_true, y_pred, y_proba):
        n_classes = len(self.classes)
        
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        
        per_class_results = []
        for i in range(n_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            
            acc = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
            prec = TP / (TP + FP) if (TP+FP) > 0 else 0
            rec = TP / (TP + FN) if (TP+FN) > 0 else 0
            spec = TN / (TN + FP) if (TN+FP) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec+rec) > 0 else 0
            err = 1 - acc
            
            per_class_results.append({
                "class": self.classes[i],
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "sensitivity": rec,
                "specificity": spec,
                "error_rate": err
            })
            
        print("\n--- Per-Class Metrics ---")
        for res in per_class_results:
            print(f"Class: {res['class']}")
            print(f"  Accuracy:    {res['accuracy']:.4f}")
            print(f"  Precision:   {res['precision']:.4f}")
            print(f"  Recall(Sens):{res['recall']:.4f}")
            print(f"  F1-Score:    {res['f1']:.4f}")
            print(f"  Specificity: {res['specificity']:.4f}")
            print(f"  Error Rate:  {res['error_rate']:.4f}")
            
        overall_acc = accuracy_score(y_true, y_pred)
        overall_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        overall_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print("\n--- Overall Metrics (Macro) ---")
        print(f"Accuracy:  {overall_acc:.4f}")
        print(f"Precision: {overall_prec:.4f}")
        print(f"Recall:    {overall_rec:.4f}")
        print(f"F1-Score:  {overall_f1:.4f}")
        
        return {
            "accuracy": overall_acc,
            "precision": overall_prec,
            "recall": overall_rec,
            "f1": overall_f1
        }

    def plot_roc_curve(self, y_true, y_proba):
        n_classes = len(self.classes)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {self.classes[i]} (area = {roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - OvR')
        plt.legend(loc="lower right")
        plt.savefig("evaluation_roc.png")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig("evaluation_confusion_matrix.png")
        plt.close()

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['f1'], label='Train F1')
        plt.plot(history['val_f1'], label='Val F1')
        plt.title('F1 Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.savefig("evaluation_training_history.png")
        plt.close()

    def print_comparison_table(self, our_metrics):
        print("\n--- Methods Comparison Table ---")
        print(f"{'Method':<15} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8}")
        print("-" * 59)
        print(f"{'BiLSTM':<15} | {'91.2%':<8} | {'90.8%':<9} | {'91.0%':<8} | {'90.9%':<8}")
        print(f"{'DBN-ResNet':<15} | {'93.1%':<8} | {'92.7%':<9} | {'92.9%':<8} | {'92.8%':<8}")
        
        acc_str = f"{our_metrics['accuracy']*100:.1f}%"
        prec_str = f"{our_metrics['precision']*100:.1f}%"
        rec_str = f"{our_metrics['recall']*100:.1f}%"
        f1_str = f"{our_metrics['f1']*100:.1f}%"
        
        print(f"{'EGNNN-GROA':<15} | {acc_str:<8} | {prec_str:<9} | {rec_str:<8} | {f1_str:<8}")
        print("-" * 59)

    def run_full_evaluation(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        our_metrics = self.compute_metrics(y_test, y_pred, y_proba)
        self.plot_roc_curve(y_test, y_proba)
        self.plot_confusion_matrix(y_test, y_pred)
        self.print_comparison_table(our_metrics)
        print("\nEvaluation completed. Plots saved to disk.")
