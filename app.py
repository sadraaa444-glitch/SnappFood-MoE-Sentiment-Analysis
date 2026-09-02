import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import re
import pickle

# ==========================================
# 1. Constants & Configurations
# ==========================================
MAX_SEQ_LEN = 50
D_MODEL = 128
D_HIDDEN = 256
NUM_EXPERTS = 4
DEVICE = 'cpu'

# ==========================================
# 2. Model Architecture Definitions
# ==========================================
class Expert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)

class SparseMoELayer(nn.Module):
    def __init__(self, d_model, d_hidden, num_experts, top_k=1, aux_loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)
        
        self.last_routing_weights = None
        self.current_aux_loss = 0.0

    def forward(self, x, return_weights=False):
        B, S, D = x.shape
        x_flat = x.view(-1, D)  
        
        router_logits = self.router(x_flat) 
        routing_weights = F.softmax(router_logits, dim=-1) 
        
        self.last_routing_weights = routing_weights.detach().view(B, S, self.num_experts)
        
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        
        f_g = routing_weights.mean(dim=0) 
        f_c = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f_c[i] = (topk_indices == i).float().mean()
        self.current_aux_loss = self.num_experts * torch.sum(f_g * f_c) * self.aux_loss_coef
        
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                token_inputs = x_flat[mask]
                expert_outputs = expert(token_inputs)
                
                weight_mask = (topk_indices == i)
                extracted_weights = topk_weights[weight_mask].unsqueeze(-1)
                
                out_flat[mask] += expert_outputs * extracted_weights
                
        out = out_flat.view(B, S, D)
        
        if return_weights:
            return out, self.last_routing_weights
        
        return out

class MoETextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, d_hidden, num_experts, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.moe = SparseMoELayer(d_model, d_hidden, num_experts, top_k=1)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, 5) 

    def forward(self, text_tokens, return_weights=False):
        B, S = text_tokens.shape
        positions = torch.arange(0, S, device=text_tokens.device).unsqueeze(0).expand(B, S)
        
        x = self.embedding(text_tokens) + self.pos_embedding(positions)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        if return_weights:
            moe_out, routing_weights = self.moe(x, return_weights=True)
        else:
            moe_out = self.moe(x)
            
        x = self.layer_norm2(x + moe_out)
        x_pooled = x.mean(dim=1) 
        logits = self.fc_out(x_pooled)
        
        if return_weights:
            seq_avg_weights = routing_weights.mean(dim=1) 
            return logits, seq_avg_weights
            
        return logits

# ==========================================
# 3. Initialization (Vocab & Model Loading)
# ==========================================
try:
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary loaded successfully. Size: {VOCAB_SIZE}")
except FileNotFoundError:
    print("WARNING: vocab.pkl not found! Please export it from your notebook.")
    # Fallback to prevent instant crash, but prediction will be garbage
    vocab = {"<PAD>": 0, "<UNK>": 1}
    VOCAB_SIZE = 2

# Initialize Model
model = MoETextClassifier(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    d_hidden=D_HIDDEN, 
    num_experts=NUM_EXPERTS, 
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

try:
    # Load trained weights (Save this from your notebook using torch.save(model.state_dict(), 'moe_model.pth'))
    model.load_state_dict(torch.load('moe_model.pth', map_location=DEVICE))
    model.eval()
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("WARNING: moe_model.pth not found! Using untrained weights.")

# ==========================================
# 4. Data Processing Utilities
# ==========================================
def clean_persian_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_pad(text, vocab_dict, max_len):
    words = text.split()
    tokens = [vocab_dict.get(word, vocab_dict.get("<UNK>", 1)) for word in words]
    if len(tokens) < max_len:
        tokens = tokens + [vocab_dict.get("<PAD>", 0)] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

# ==========================================
# 5. FastAPI Application
# ==========================================
app = FastAPI(title="SnappFood MoE Sentiment API")
templates = Jinja2Templates(directory="templates")

class InferenceRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"num_experts": NUM_EXPERTS}
)

@app.post("/api/predict")
async def predict_sentiment(request: InferenceRequest):
    raw_text = request.text
    
    try:
        # Preprocess & Tokenize
        cleaned_text = clean_persian_text(raw_text)
        tokens = tokenize_and_pad(cleaned_text, vocab, MAX_SEQ_LEN)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            logits, routing_weights = model(tokens_tensor, return_weights=True)
            
        # Extract Results
        predicted_class = torch.argmax(logits, dim=-1).item() + 1 # +1 because labels are 1-5
        weights_list = routing_weights[0].tolist() 
        
        return {
            "prediction": f"{predicted_class} ستاره",
            "routing_weights": weights_list,
            "status": "success"
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)