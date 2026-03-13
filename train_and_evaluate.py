import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import os


class ComparativeLearning:
    """BERT-based contrastive learning for anomaly detection using sentence embeddings."""

    def __init__(self, train_sentences, train_negative_sentences, test_sentences, test_true, save_path, result_path):
        self.train_sentences = train_sentences
        self.train_negative_sentences = train_negative_sentences
        self.test_sentences = test_sentences
        self.test_true = test_true
        self.save_path = save_path
        self.result_path = result_path

        # Initialize independent BERT model and tokenizer for each instance
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.train()

    def get_sentence_embedding(self, sentence):
        """Compute sentence embedding using BERT mean pooling."""
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def nt_xent_loss(self, anchor_embedding, positive_embedding, negative_embeddings, temperature=0.5):
        """Compute NT-Xent (SimCLR) contrastive loss."""
        anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
        positive_embedding = F.normalize(positive_embedding, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

        pos_sim = torch.sum(anchor_embedding * positive_embedding, dim=1) / temperature
        neg_sims = torch.mm(anchor_embedding, negative_embeddings.t()) / temperature

        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sims).sum(dim=1)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        return loss.mean()

    def train_contrastive_model(self, epochs=20, temperature=0.5, weight_decay=1e-5, batch_size=16, save_path=None):
        """Train BERT with contrastive learning using dropout augmentation."""
        losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        f = open(self.result_path + '/training_loss.txt', 'w')

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            total_loss = 0
            num_batches = (len(self.train_sentences) + batch_size - 1) // batch_size

            for i in range(0, len(self.train_sentences), batch_size):
                batch_pos = self.train_sentences[i:i+batch_size]
                batch_neg = self.train_negative_sentences[i:i+batch_size]

                optimizer.zero_grad()

                # Encode each positive sentence twice (dropout produces different embeddings)
                anchor_embeddings = []
                positive_embeddings = []
                negative_embeddings = []

                for pos_sent, neg_sent in zip(batch_pos, batch_neg):
                    anchor_emb = self.get_sentence_embedding(pos_sent)
                    positive_emb = self.get_sentence_embedding(pos_sent)
                    negative_emb = self.get_sentence_embedding(neg_sent)

                    anchor_embeddings.append(anchor_emb)
                    positive_embeddings.append(positive_emb)
                    negative_embeddings.append(negative_emb)

                anchor_embeddings = torch.stack(anchor_embeddings)
                positive_embeddings = torch.stack(positive_embeddings)
                negative_embeddings = torch.stack(negative_embeddings)

                # Use the second encoding of the same sentence as the positive pair
                batch_loss = 0
                for j in range(len(anchor_embeddings)):
                    anchor = anchor_embeddings[j:j+1]
                    positive = positive_embeddings[j:j+1]
                    negatives = negative_embeddings

                    loss = self.nt_xent_loss(anchor, positive, negatives, temperature)
                    batch_loss += loss

            batch_loss = batch_loss / len(anchor_embeddings)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += batch_loss.item()

            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            f.write(str(avg_loss) + "\n")

            scheduler.step(avg_loss)

        f.close()

        if save_path is not None:
            self.save_model(save_path)
            print(f"Model saved to: {save_path}")

        return losses

    def save_model(self, save_path):
        """Save the fine-tuned BERT model and tokenizer."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

    def detect_anomalies(self):
        """Detect anomalies using cosine similarity with training embeddings."""
        scores = []
        f = open(self.result_path + '/scores.txt', 'w')

        print("Pre-computing training embeddings...")
        train_embeddings = []
        with torch.no_grad():
            for train_sentence in tqdm(self.train_sentences, desc="Computing train embeddings"):
                embedding = self.get_sentence_embedding(train_sentence)
                train_embeddings.append(embedding)
        train_embeddings = torch.stack(train_embeddings)

        for sentence in tqdm(self.test_sentences, desc="Detecting anomalies"):
            with torch.no_grad():
                embedding = self.get_sentence_embedding(sentence)
                embedding = embedding.unsqueeze(0)

                similarities = F.cosine_similarity(embedding, train_embeddings, dim=1)
                similarities = (similarities + 1) / 2

                mean_similarity = similarities.mean().item()
                scores.append(mean_similarity)
                f.write(str(mean_similarity) + "\n")
        f.close()

        return scores

    def detect_anomalies_improved(self):
        """Detect anomalies using KNN-based distance scoring in eval mode."""
        scores = []
        f = open(self.result_path + '/scores.txt', 'w')

        # Switch to eval mode (disables dropout)
        self.model.eval()

        print("Pre-computing training embeddings...")
        train_embeddings = []
        with torch.no_grad():
            for train_sentence in tqdm(self.train_sentences, desc="Computing train embeddings"):
                embedding = self.get_sentence_embedding(train_sentence)
                train_embeddings.append(embedding)
        train_embeddings = torch.stack(train_embeddings)
        train_embeddings = F.normalize(train_embeddings, p=2, dim=1)

        for sentence in tqdm(self.test_sentences, desc="Detecting anomalies"):
            with torch.no_grad():
                embedding = self.get_sentence_embedding(sentence)
                embedding = embedding.unsqueeze(0)
                embedding = F.normalize(embedding, p=2, dim=1)

                # KNN-based anomaly score: mean distance to k nearest neighbors
                distances = torch.cdist(embedding, train_embeddings, p=2).squeeze(0)
                top_k = min(5, len(distances))
                knn_mean_distance = distances.topk(top_k, largest=False).values.mean().item()
                knn_score = 1.0 / (1.0 + knn_mean_distance)

                scores.append(knn_score)
                f.write(f"{knn_mean_distance},{knn_score}\n")

        # Restore training mode for subsequent training
        self.model.train()

        f.close()
        return scores

    def train_and_evaluate_model(self, save_model_path=None):
        """Train the model and evaluate using ROC-AUC."""
        losses = self.train_contrastive_model(save_path=save_model_path)

        # Plot training loss
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(self.result_path + '/training_loss.png')

        scores = self.detect_anomalies_improved()

        # Calculate and plot ROC curve
        fpr, tpr, thresholds = roc_curve(self.test_true, scores)
        auc_score = roc_auc_score(self.test_true, scores)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.6f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(self.result_path + '/sklearn_roc_curve.png')
        print("roc_auc score is ")
        print(roc_auc_score(self.test_true, scores))

        return auc_score
