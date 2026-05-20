"""
KLUE-RE Fine-tuning: klue/roberta-base
Entity Marker approach:
  [E1]subject[/E1] context [E2]object[/E2]
  → hidden state at [E1] pos + hidden state at [E2] pos → classifier

Reference: Soares et al. (2019) "Matching the Blanks: Distributional Similarity for RE"
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

os.makedirs("docs", exist_ok=True)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

MODEL_NAME = 'klue/roberta-base'
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 2e-5
WARMUP_RATIO = 0.1


# ─────────────────────────────────────────────────────────────────────
# Tokenizer & special tokens
# ─────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
SPECIAL_TOKENS = ['[E1]', '[/E1]', '[E2]', '[/E2]']
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

E1_ID = tokenizer.convert_tokens_to_ids('[E1]')
E2_ID = tokenizer.convert_tokens_to_ids('[E2]')


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
class KLUEREDataset(Dataset):
    def __init__(self, df, label2idx, max_len=MAX_LEN):
        self.samples = []
        for _, row in df.iterrows():
            enc = tokenizer(
                row['marked_text'],
                max_length=max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            ids  = enc['input_ids'].squeeze(0)
            mask = enc['attention_mask'].squeeze(0)

            # [E1], [E2] 토큰 위치 탐색
            e1_pos = (ids == E1_ID).nonzero(as_tuple=True)[0]
            e2_pos = (ids == E2_ID).nonzero(as_tuple=True)[0]
            e1_pos = int(e1_pos[0]) if len(e1_pos) > 0 else 0
            e2_pos = int(e2_pos[0]) if len(e2_pos) > 0 else 0

            self.samples.append({
                'input_ids':      ids,
                'attention_mask': mask,
                'e1_pos': e1_pos,
                'e2_pos': e2_pos,
                'label':  label2idx[row['final_relation']],
            })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'e1_pos':         torch.tensor([b['e1_pos']        for b in batch]),
        'e2_pos':         torch.tensor([b['e2_pos']        for b in batch]),
        'label':          torch.tensor([b['label']         for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────
# Model: [E1] hidden + [E2] hidden → MLP → label
# ─────────────────────────────────────────────────────────────────────
class KLUEREModel(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.encoder.resize_token_embeddings(len(tokenizer))
        H = self.encoder.config.hidden_size          # 768

        self.classifier = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, num_labels),
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        out   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                # (B, L, H)

        B = hidden.size(0)
        e1_h = hidden[torch.arange(B, device=hidden.device), e1_pos]  # (B, H)
        e2_h = hidden[torch.arange(B, device=hidden.device), e2_pos]  # (B, H)

        return self.classifier(torch.cat([e1_h, e2_h], dim=1))        # (B, num_labels)


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────
def run_finetune():
    from klue_data_loader import load_klue_re

    print("KLUE-RE 데이터 로딩...")
    train_df = load_klue_re('train')
    val_df   = load_klue_re('validation')

    unique_labels = sorted(train_df['final_relation'].unique())
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    num_labels = len(unique_labels)
    print(f"  Labels: {num_labels}  |  Train: {len(train_df)}  |  Val: {len(val_df)}")

    # 클래스 불균형 가중치
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(unique_labels),
        y=train_df['final_relation'].tolist()
    )

    device = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available()         else 'cpu'
    )
    print(f"  Device: {device}")

    print("  Dataset 구축 중 (tokenizing)...")
    t0 = time.time()
    train_ds = KLUEREDataset(train_df, label2idx)
    val_ds   = KLUEREDataset(val_df,   label2idx)
    print(f"  Tokenize 완료: {time.time()-t0:.0f}s")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=0)

    model = KLUEREModel(num_labels).to(device)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(
        [{'params': model.encoder.parameters(),    'lr': LR},
         {'params': model.classifier.parameters(), 'lr': LR * 5}],
        weight_decay=0.01
    )
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    train_losses, val_f1s = [], []
    best_f1, best_epoch = 0.0, 0

    print(f"\n학습 시작 (총 {EPOCHS} epoch, {total_steps} steps)...")
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────
        model.train()
        epoch_loss = 0
        t_ep = time.time()
        for step, batch in enumerate(train_loader):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            e1   = batch['e1_pos'].to(device)
            e2   = batch['e2_pos'].to(device)
            lbl  = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask, e1, e2)
            loss   = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if (step + 1) % 200 == 0:
                print(f"  Epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={epoch_loss/(step+1):.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ── Eval ───────────────────────────────────────
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                e1   = batch['e1_pos'].to(device)
                e2   = batch['e2_pos'].to(device)
                logits = model(ids, mask, e1, e2)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(batch['label'].numpy())

        macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        val_f1s.append(macro_f1)

        elapsed = time.time() - t_ep
        print(f"  ▶ Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Macro F1: {macro_f1:.4f} | {elapsed:.0f}s")

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), 'docs/klue_roberta_best.pt')
            print(f"    ✅ Best model saved (epoch {epoch})")

    print(f"\n최고 Val Macro F1: {best_f1:.4f} (epoch {best_epoch})")

    # ── Load best & final evaluation ──────────────────
    model.load_state_dict(torch.load('docs/klue_roberta_best.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            e1   = batch['e1_pos'].to(device)
            e2   = batch['e2_pos'].to(device)
            logits = model(ids, mask, e1, e2)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(batch['label'].numpy())

    final_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    print(f"Best checkpoint Macro F1: {final_f1:.4f}")

    # ── 시각화 1: Training Curve ───────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, EPOCHS+1), train_losses, 'b-o', linewidth=2, label='Train Loss')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"klue/roberta-base — Training Curve\n(class-weighted, entity marker)", fontweight='bold')
    ax1.grid(alpha=0.3); sns.despine(ax=ax1)

    ax2.plot(range(1, EPOCHS+1), val_f1s, 'g-s', linewidth=2, label='Val Macro F1')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_ylim(0, 1)
    ax2.set_title(f"klue/roberta-base — Validation Macro F1\nBest: {best_f1:.4f} @ epoch {best_epoch}",
                  fontweight='bold')
    ax2.grid(alpha=0.3); sns.despine(ax=ax2)
    plt.suptitle("KLUE-RE Fine-tuning: klue/roberta-base", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("docs/klue_plm_training.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ docs/klue_plm_training.png")

    # ── 시각화 2: Confusion Matrix (top 15) ───────────
    pred_labels = [idx2label[p] for p in preds]
    true_labels = [idx2label[t] for t in trues]
    top15 = pd.Series(true_labels).value_counts().head(15).index.tolist()
    mask  = np.isin(true_labels, top15)
    cm = confusion_matrix(
        np.array(true_labels)[mask],
        np.array(pred_labels)[mask],
        labels=top15
    )
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top15, yticklabels=top15)
    plt.title(f"KLUE-RE klue/roberta-base — Confusion Matrix\n(Macro F1={final_f1:.4f}, Top-15 relations)",
              fontsize=13, fontweight='bold')
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("docs/klue_plm_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ docs/klue_plm_confusion_matrix.png")

    # ── 시각화 3: 방법론별 KLUE 성능 비교 ─────────────
    methods = ['Pattern\n(TF-IDF)', 'Embedding\n(SBERT)', 'Feature\nRF',
               'Kernel\nSVM', 'Bi-LSTM\n+Attn', 'klue/\nroBERTa']
    scores  = [0.0897, 0.1392, 0.1626, 0.2222, 0.0706, final_f1]
    colors  = ['#ffb3b3','#99ccff','#b3ffb3','#ffe599','#d5b3ff','#ff7f0e']

    plt.figure(figsize=(11, 6))
    bars = plt.bar(methods, scores, color=colors, edgecolor='#666', linewidth=0.8, width=0.55)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.008,
                 f"{h:.4f}", ha='center', fontsize=11, fontweight='bold')
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='0.5 baseline')
    plt.ylabel("Macro F1 / V-Measure", fontsize=11)
    plt.ylim(0, min(final_f1 * 1.2, 1.0))
    plt.title("KLUE-RE — All Methods Comparison\n(Macro F1, Validation Set 7,765)",
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig("docs/klue_final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ docs/klue_final_comparison.png (updated)")

    # ── 결과 저장 ─────────────────────────────────────
    results = {}
    if os.path.exists('docs/klue_results.json'):
        with open('docs/klue_results.json') as f:
            results = json.load(f)
    results['klue_plm_roberta'] = round(final_f1, 4)
    with open('docs/klue_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ docs/klue_results.json  PLM F1={final_f1:.4f}")
    return final_f1


if __name__ == "__main__":
    t0 = time.time()
    f1 = run_finetune()
    print(f"\n⏱  Total: {(time.time()-t0)/60:.1f} min")
    print(f"🎉 KLUE-RE klue/roberta-base Macro F1: {f1:.4f}")
