"""
Step 6: klue/roberta-large Fine-tuning on OIA Domain (v3)
──────────────────────────────────────────────────────────
개선 사항:
  1) Entity Type Embedding — head/tail 타입을 추가 feature로 활용
  2) [CLS] + [E1] + [E2] concat — 전체 문장 표현 추가
  3) Focal Loss — 소수 클래스 강화
  4) klue/roberta-large — base → large 업그레이드
  5) R-Drop — 동일 샘플 2회 forward, KL divergence로 정규화
"""

import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

os.makedirs("docs", exist_ok=True)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME   = 'klue/roberta-large'
MAX_LEN      = 128
BATCH_SIZE   = 8          # large 모델이라 메모리 절약
EPOCHS       = 15
LR           = 1e-5       # large는 lr을 낮게
WARMUP_RATIO = 0.1
DROPOUT      = 0.1
PATIENCE     = 4
FOCAL_GAMMA  = 2.0        # Focal Loss gamma
TYPE_EMB_DIM = 32         # Entity type embedding 차원
RDROP_ALPHA  = 0.5        # R-Drop KL divergence 가중치
SEED         = 42
CORPUS_PATH  = 'data/re_fixed_v6/corpus_clean.jsonl'
SAVE_PATH    = 'docs/oia_roberta_best.pt'

# ── 1. 데이터 로드 & 분할 ────────────────────────────────────────────────────
def load_oia_corpus(path=CORPUS_PATH):
    with open(path, encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    rows = []
    for d in data:
        marked   = d.get('marked_text') or d.get('sentence', '')
        relation = d.get('relation', 'NO_RELATION').strip().upper()
        head_type = d.get('head', {}).get('type', 'UNKNOWN')
        tail_type = d.get('tail', {}).get('type', 'UNKNOWN')
        rows.append({
            'marked_text': marked,
            'relation':    relation,
            'head_type':   head_type,
            'tail_type':   tail_type,
        })

    df = pd.DataFrame(rows)
    print(f"✅ OIA 코퍼스 로드: {len(df)}건, {df['relation'].nunique()}개 관계")
    print(f"   관계 분포:\n{df['relation'].value_counts().to_string()}\n")
    return df


def split_data(df, val_ratio=0.1, test_ratio=0.15):
    counts    = df['relation'].value_counts()
    rare      = counts[counts < 10].index.tolist()
    rare_df   = df[df['relation'].isin(rare)]
    normal_df = df[~df['relation'].isin(rare)]

    np.random.seed(SEED)
    train_n, val_n = train_test_split(
        normal_df, test_size=val_ratio + test_ratio,
        stratify=normal_df['relation'], random_state=SEED
    )
    val_n, test_n = train_test_split(
        val_n,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=val_n['relation'], random_state=SEED
    )

    train_df = pd.concat([train_n, rare_df], ignore_index=True)
    val_df   = val_n.reset_index(drop=True)
    test_df  = test_n.reset_index(drop=True)

    print(f"✅ 분할 완료 — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    if rare:
        print(f"   소수 클래스 → train 전량 배치: {rare}")
    return train_df, val_df, test_df


def build_type_vocab(df):
    """corpus 전체에서 entity type 어휘 구축"""
    types = set(df['head_type'].tolist()) | set(df['tail_type'].tolist())
    types.add('UNKNOWN')
    type2idx = {t: i for i, t in enumerate(sorted(types))}
    return type2idx


# ── 2. Tokenizer & Dataset ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
SPECIAL_TOKENS = ['[E1]', '[/E1]', '[E2]', '[/E2]']
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
E1_ID = tokenizer.convert_tokens_to_ids('[E1]')
E2_ID = tokenizer.convert_tokens_to_ids('[E2]')


class OIADataset(Dataset):
    def __init__(self, df, label2idx, type2idx, max_len=MAX_LEN):
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

            e1_pos_list = (ids == E1_ID).nonzero(as_tuple=True)[0]
            e2_pos_list = (ids == E2_ID).nonzero(as_tuple=True)[0]
            e1_pos = int(e1_pos_list[0]) if len(e1_pos_list) > 0 else 0
            e2_pos = int(e2_pos_list[0]) if len(e2_pos_list) > 0 else 0

            head_type_idx = type2idx.get(row['head_type'], type2idx['UNKNOWN'])
            tail_type_idx = type2idx.get(row['tail_type'], type2idx['UNKNOWN'])

            self.samples.append({
                'input_ids':      ids,
                'attention_mask': mask,
                'e1_pos':         e1_pos,
                'e2_pos':         e2_pos,
                'head_type':      head_type_idx,
                'tail_type':      tail_type_idx,
                'label':          label2idx[row['relation']],
            })

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'e1_pos':         torch.tensor([b['e1_pos']        for b in batch]),
        'e2_pos':         torch.tensor([b['e2_pos']        for b in batch]),
        'head_type':      torch.tensor([b['head_type']     for b in batch]),
        'tail_type':      torch.tensor([b['tail_type']     for b in batch]),
        'label':          torch.tensor([b['label']         for b in batch]),
    }


# ── 3. Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=FOCAL_GAMMA):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── 4. 모델: [CLS]+[E1]+[E2] + Entity Type Embedding ─────────────────────────
class OIAREModel(nn.Module):
    def __init__(self, num_labels, num_types, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.encoder.resize_token_embeddings(len(tokenizer))
        H = self.encoder.config.hidden_size  # 768

        # [방법 1] Entity type embedding
        self.type_embedding = nn.Embedding(num_types, TYPE_EMB_DIM)

        # [방법 2] [CLS]+[E1]+[E2] → H*3,  head/tail type → TYPE_EMB_DIM*2
        in_dim = H * 3 + TYPE_EMB_DIM * 2

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, num_labels),
        )

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos, head_type, tail_type):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state   # (B, L, H)
        B      = hidden.size(0)
        device = hidden.device

        cls_h = hidden[:, 0, :]                                          # (B, H)
        e1_h  = hidden[torch.arange(B, device=device), e1_pos]          # (B, H)
        e2_h  = hidden[torch.arange(B, device=device), e2_pos]          # (B, H)

        # Entity type embeddings
        ht_emb = self.type_embedding(head_type)   # (B, TYPE_EMB_DIM)
        tt_emb = self.type_embedding(tail_type)   # (B, TYPE_EMB_DIM)

        concat = torch.cat([cls_h, e1_h, e2_h, ht_emb, tt_emb], dim=1)
        return self.classifier(concat)


# ── 5. R-Drop KL divergence ──────────────────────────────────────────────────
def compute_kl_loss(p, q):
    """Symmetric KL divergence between two logit distributions."""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2


# ── 6. 학습 / 평가 루프 ──────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, preds, trues = 0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            e1   = batch['e1_pos'].to(device)
            e2   = batch['e2_pos'].to(device)
            ht   = batch['head_type'].to(device)
            tt   = batch['tail_type'].to(device)
            lbl  = batch['label'].to(device)

            if train:
                optimizer.zero_grad()
                # R-Drop: 동일 배치를 2번 forward (dropout mask 다름)
                logits1 = model(ids, mask, e1, e2, ht, tt)
                logits2 = model(ids, mask, e1, e2, ht, tt)
                cls_loss = (criterion(logits1, lbl) + criterion(logits2, lbl)) / 2
                kl_loss  = compute_kl_loss(logits1, logits2)
                loss     = cls_loss + RDROP_ALPHA * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                logits = (logits1 + logits2) / 2  # 앙상블 예측
            else:
                logits = model(ids, mask, e1, e2, ht, tt)
                loss   = criterion(logits, lbl)

            total_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(lbl.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    return avg_loss, macro_f1, preds, trues


# ── 6. 메인 ──────────────────────────────────────────────────────────────────
def run_finetune():
    print("=== Step 6. klue/roberta-large Fine-tuning on OIA (v3) ===")
    print("    개선: Entity Type Emb + [CLS]+[E1]+[E2] + Focal Loss + R-Drop\n")

    df = load_oia_corpus()
    train_df, val_df, test_df = split_data(df)

    all_labels = sorted(df['relation'].unique())
    label2idx  = {l: i for i, l in enumerate(all_labels)}
    idx2label  = {i: l for l, i in label2idx.items()}
    num_labels = len(all_labels)
    print(f"✅ 관계 수: {num_labels}개")

    type2idx  = build_type_vocab(df)
    num_types = len(type2idx)
    print(f"✅ Entity type 수: {num_types}개 → {sorted(type2idx.keys())}\n")

    # 클래스 가중치 (Focal Loss에도 적용)
    train_rels    = train_df['relation'].tolist()
    class_weights = compute_class_weight(
        'balanced', classes=np.array(all_labels),
        y=[r for r in train_rels if r in all_labels]
    )

    device = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available()         else 'cpu'
    )
    print(f"✅ Device: {device}")

    print("\n토크나이징 중...")
    t0 = time.time()
    train_ds = OIADataset(train_df, label2idx, type2idx)
    val_ds   = OIADataset(val_df,   label2idx, type2idx)
    test_ds  = OIADataset(test_df,  label2idx, type2idx)
    print(f"완료: {time.time()-t0:.1f}s\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = OIAREModel(num_labels, num_types).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=weight_tensor, gamma=FOCAL_GAMMA)

    optimizer = torch.optim.AdamW(
        [{'params': model.encoder.parameters(),       'lr': LR},
         {'params': model.type_embedding.parameters(),'lr': LR * 10},
         {'params': model.classifier.parameters(),    'lr': LR * 10}],
        weight_decay=0.01
    )
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
    best_val_f1, patience_cnt = 0.0, 0

    print(f"학습 시작 (EPOCHS={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, gamma={FOCAL_GAMMA}, rdrop_alpha={RDROP_ALPHA})")
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, train=True)
        va_loss, va_f1, _, _ = run_epoch(model, val_loader,   optimizer, scheduler, criterion, device, train=False)

        history['train_loss'].append(tr_loss)
        history['train_f1'].append(tr_f1)
        history['val_loss'].append(va_loss)
        history['val_f1'].append(va_f1)

        tag = ''
        if va_f1 > best_val_f1:
            best_val_f1  = va_f1
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_PATH)
            tag = ' ✓ best'
        else:
            patience_cnt += 1
            tag = f' (patience {patience_cnt}/{PATIENCE})'

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"TrLoss {tr_loss:.4f} TrF1 {tr_f1:.4f} | "
              f"VaLoss {va_loss:.4f} VaF1 {va_f1:.4f} | "
              f"{time.time()-t_ep:.0f}s{tag}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 최적 모델 테스트 평가
    print(f"\n최적 모델 로드 (Best Val F1: {best_val_f1:.4f})")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    _, test_f1, test_preds, test_trues = run_epoch(
        model, test_loader, optimizer, scheduler, criterion, device, train=False
    )

    pred_labels = [idx2label[p] for p in test_preds]
    true_labels = [idx2label[t] for t in test_trues]

    print(f"\n▶ Test Macro F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    # ── 시각화 1: Training Curve ──────────────────────────────────────────────
    ep_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ep_range, history['train_loss'], 'b-o', lw=2, label='Train Loss')
    ax1.plot(ep_range, history['val_loss'],   'r-s', lw=2, label='Val Loss')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss")
    ax1.set_title("OIA roberta-base v2 — Loss Curve")
    ax1.legend(); ax1.grid(alpha=0.3); sns.despine(ax=ax1)

    ax2.plot(ep_range, history['train_f1'], 'b-o', lw=2, label='Train F1')
    ax2.plot(ep_range, history['val_f1'],   'r-s', lw=2, label='Val F1')
    ax2.axhline(test_f1, color='green', ls='--', lw=1.5, label=f'Test F1={test_f1:.4f}')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1"); ax2.set_ylim(0, 1)
    ax2.set_title(f"OIA roberta-base v2 — F1\nBest Val={best_val_f1:.4f}")
    ax2.legend(); ax2.grid(alpha=0.3); sns.despine(ax=ax2)

    plt.tight_layout()
    plt.savefig("docs/step6_training_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/step6_training_curve.png")

    # ── 시각화 2: 방법론 비교 ─────────────────────────────────────────────────
    methods = ['Bi-LSTM\n+Attn', 'Feature\nRF', 'Kernel\nSVM',
               'roBERTa-base\n(OIA)', 'roBERTa-large\n+R-Drop']
    scores  = [0.3624,           0.73,           0.8627,      0.8807,            test_f1]
    colors  = ['#d5b3ff',        '#b3ffb3',      '#ffe599',   '#ffb347',         '#ff7f0e']

    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, scores, color=colors, edgecolor='#555', lw=0.8, width=0.5)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.01,
                 f"{h:.4f}", ha='center', fontsize=11, fontweight='bold')
    plt.ylabel("Macro F1", fontsize=11)
    plt.ylim(0, 1.05)
    plt.title("OIA Domain — 방법론별 Macro F1 비교 (v2 개선)", fontsize=13, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    plt.savefig("docs/step6_final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/step6_final_comparison.png")

    # ── 시각화 3: Confusion Matrix ────────────────────────────────────────────
    labels_in_test = sorted(set(true_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_in_test)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_in_test, yticklabels=labels_in_test)
    plt.title(f"OIA roberta-base v2 — Confusion Matrix (Test F1={test_f1:.4f})",
              fontsize=12, fontweight='bold')
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig("docs/step6_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/step6_confusion_matrix.png")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    results = {}
    if os.path.exists('docs/results.json'):
        with open('docs/results.json') as f:
            results = json.load(f)
    results['dl_roberta_oia_v3'] = round(test_f1, 4)
    with open('docs/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*52}")
    print(f"  BiLSTM+Attn              :  0.3624")
    print(f"  roBERTa-base (OIA)       :  0.8807")
    print(f"  roBERTa-large+R-Drop     : {test_f1:.4f}  ← Step 6 v3")
    print(f"  Kernel SVM               :  0.8627")
    print(f"{'='*52}")
    return test_f1


if __name__ == "__main__":
    t0 = time.time()
    f1 = run_finetune()
    print(f"\n⏱  Total: {(time.time()-t0)/60:.1f} min")
    print(f"🎉 OIA klue/roberta-large v3 Macro F1: {f1:.4f}")
