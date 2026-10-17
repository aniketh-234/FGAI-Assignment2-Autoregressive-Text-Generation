"""ICT4403 Assignment 2: Autoregressive text generation using RNN, BiRNN, LSTM, GRU.
Run: pip install torch pandas numpy matplotlib requests; python autoregressive_text_generation.py
Dataset downloads from the URL below. Results are saved to ./results.
"""
import re, time, math, random, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT=Path('results'); OUT.mkdir(exist_ok=True)
DATA_URL='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
CONTEXT_LEN=10; STRIDE=2; BATCH_SIZE=128; MAX_VOCAB=8000
EMBED_DIM=64; HIDDEN_DIM=96; DROPOUT=0.15; EPOCHS=3

def tokenize(text): return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?,;:]", text.lower())

print('Device:', DEVICE)
response=requests.get(DATA_URL, timeout=30); response.raise_for_status(); raw_text=response.text
all_tokens=tokenize(raw_text); n=len(all_tokens)
train_end=int(n*.8); val_end=int(n*.9)
train_tokens=all_tokens[:train_end]; val_tokens=all_tokens[train_end:val_end]; test_tokens=all_tokens[val_end:]
counts=Counter(train_tokens); vocab=['<unk>']+[w for w,_ in counts.most_common(MAX_VOCAB-1)]
stoi={w:i for i,w in enumerate(vocab)}; itos={i:w for w,i in stoi.items()}; UNK=stoi['<unk>']
def encode(tokens): return [stoi.get(t,UNK) for t in tokens]
train_ids,val_ids,test_ids=map(encode,(train_tokens,val_tokens,test_tokens))
print(f'Tiny Shakespeare tokens: total={n:,}, train={len(train_ids):,}, val={len(val_ids):,}, test={len(test_ids):,}; vocab={len(vocab)}')

class NextWordDataset(Dataset):
    def __init__(self,ids,context_len=CONTEXT_LEN,stride=STRIDE):
        self.ids=ids; self.context_len=context_len; self.starts=list(range(0,len(ids)-context_len,stride))
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=self.starts[i]; return torch.tensor(self.ids[s:s+self.context_len],dtype=torch.long),torch.tensor(self.ids[s+self.context_len],dtype=torch.long)
train_loader=DataLoader(NextWordDataset(train_ids),batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(NextWordDataset(val_ids,stride=1),batch_size=BATCH_SIZE)
test_loader=DataLoader(NextWordDataset(test_ids,stride=1),batch_size=BATCH_SIZE)

class NextWordModel(nn.Module):
    def __init__(self,kind,vocab_size):
        super().__init__(); self.kind=kind; self.embedding=nn.Embedding(vocab_size,EMBED_DIM)
        if kind=='Simple RNN': self.rnn=nn.RNN(EMBED_DIM,HIDDEN_DIM,batch_first=True); out=HIDDEN_DIM
        elif kind=='Bidirectional RNN': self.rnn=nn.RNN(EMBED_DIM,HIDDEN_DIM,batch_first=True,bidirectional=True); out=2*HIDDEN_DIM
        elif kind=='LSTM': self.rnn=nn.LSTM(EMBED_DIM,HIDDEN_DIM,batch_first=True); out=HIDDEN_DIM
        elif kind=='GRU': self.rnn=nn.GRU(EMBED_DIM,HIDDEN_DIM,batch_first=True); out=HIDDEN_DIM
        else: raise ValueError(kind)
        self.dropout=nn.Dropout(DROPOUT); self.fc=nn.Linear(out,vocab_size)
    def forward(self,x):
        y=self.rnn(self.embedding(x))[0]
        return self.fc(self.dropout(y[:,-1,:]))

def evaluate(model,loader):
    model.eval(); loss_fn=nn.CrossEntropyLoss(); total_loss=correct=count=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE); logits=model(x); loss=loss_fn(logits,y); b=len(y)
            total_loss+=loss.item()*b; correct+=(logits.argmax(1)==y).sum().item(); count+=b
    loss=total_loss/max(count,1)
    return {'loss':loss,'perplexity':math.exp(min(loss,20)),'accuracy':correct/max(count,1)}

def train_model(kind):
    model=NextWordModel(kind,len(vocab)).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=.002); loss_fn=nn.CrossEntropyLoss(); hist=[]; start=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); total=seen=0
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(y); seen+=len(y)
        vm=evaluate(model,val_loader); hist.append({'epoch':ep,'train_loss':total/max(seen,1),'val_loss':vm['loss'],'val_perplexity':vm['perplexity'],'val_accuracy':vm['accuracy']})
        print(f'{kind} epoch {ep}/{EPOCHS}: train loss={total/max(seen,1):.4f}, val ppl={vm["perplexity"]:.2f}, val acc={vm["accuracy"]:.4f}')
    elapsed=time.time()-start; tm=evaluate(model,test_loader)
    row={'model':kind,'test_loss':tm['loss'],'test_perplexity':tm['perplexity'],'test_accuracy':tm['accuracy'],'training_time_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters())}
    torch.save(model.state_dict(),OUT/(kind.lower().replace(' ','_')+'.pt'))
    return model,hist,row

names=['Simple RNN','Bidirectional RNN','LSTM','GRU']; models={}; histories={}; rows=[]
for name in names:
    print('\n'+'='*60+'\nTraining '+name)
    model,hist,row=train_model(name); models[name]=model; histories[name]=hist; rows.append(row)
results=pd.DataFrame(rows).sort_values('test_perplexity'); results.to_csv(OUT/'model_comparison.csv',index=False); print('\nRESULTS\n',results.to_string(index=False))
plt.figure(figsize=(9,5))
for name,hist in histories.items(): plt.plot([x['epoch'] for x in hist],[x['train_loss'] for x in hist],marker='o',label=name)
plt.xlabel('Epoch'); plt.ylabel('Training cross-entropy loss'); plt.title('Training Loss by Model'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'training_loss.png',dpi=160); plt.close()
plt.figure(figsize=(9,5)); plt.bar(results['model'],results['test_perplexity']); plt.ylabel('Test perplexity (lower is better)'); plt.title('Test Perplexity Comparison'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(OUT/'test_perplexity.png',dpi=160); plt.close()

def generate(model,seed_text,temperature,max_new_tokens=50):
    model.eval(); seed=tokenize(seed_text); context=[stoi.get(t,UNK) for t in seed]; generated=list(seed)
    for _ in range(max_new_tokens):
        ids=context[-CONTEXT_LEN:]
        if len(ids)<CONTEXT_LEN: ids=[UNK]*(CONTEXT_LEN-len(ids))+ids
        x=torch.tensor([ids],dtype=torch.long,device=DEVICE)
        with torch.no_grad(): logits=model(x)[0]
        probs=torch.softmax(logits/max(float(temperature),1e-5),dim=-1); nxt=int(torch.multinomial(probs,1).item())
        generated.append(itos[nxt]); context.append(nxt)
    return re.sub(r'\s+([.!?,;:])',r'\1',' '.join(generated))

seeds=['the king','i will not','what is the']; temps=[.5,1.0,1.5]; samples=[]
for name,model in models.items():
    for seed in seeds:
        for temp in temps:
            txt=generate(model,seed,temp); samples.append({'model':name,'seed':seed,'temperature':temp,'generated_text':txt})
            print(f'\n[{name} | seed={seed!r} | temperature={temp}]\n{txt}')
samples_df=pd.DataFrame(samples)
def repetition_ratio(text):
    t=tokenize(text); bg=list(zip(t,t[1:])); return 0.0 if len(bg)<2 else 1-len(set(bg))/len(bg)
samples_df['repetition_ratio']=samples_df.generated_text.apply(repetition_ratio)
samples_df.to_csv(OUT/'generated_samples.csv',index=False)
summary={'dataset':'Tiny Shakespeare','dataset_source':DATA_URL,'context_length':CONTEXT_LEN,'vocabulary_size':len(vocab),'epochs':EPOCHS,'results':results.to_dict(orient='records'),'best_by_test_perplexity':str(results.iloc[0]['model'])}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('\nSaved model_comparison.csv, generated_samples.csv, summary.json and plots in',OUT.resolve())
print('Qualitative evaluation: inspect samples at all temperatures and rate coherence, diversity, contextual consistency, and repetition before selecting the overall best model.')
