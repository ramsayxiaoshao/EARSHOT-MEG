import os, math, pickle, random, time, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# =========================
# 路径与配置
# =========================

@dataclass
class Config:
    # 数据根：这层里要有 Burgundy/、Speakers.txt、MALD-NEIGHBORS-1000.txt、GAMMATONE_64_100/
    DATA_ROOT: Path = Path("D:/00CODE00/dataset/Dataset/EARSHOT-gammatone+/")  # <-- 改成你的
    GAMMATONE_DIRNAME: str = "GAMMATONE_64_100"            # 64 bands @100Hz
    LEXICON_TXT: str = "MALD-NEIGHBORS-1000.txt"           # 2934 词
    SPEAKERS_TXT: str = "Speakers.txt"
    USE_SET: str = "train"   # 'train' / 'test' / 'even'（简化：默认 train）
    # 生成参数（论文 STIMULUS_SEQUENCES['2']）
    N_WORDS: int = 2
    N_PHRASES: int = 2
    MIN_SIL: int = 20   # 帧 = 200ms
    MAX_SIL: int = 50   # 帧 = 500ms
    SEGMENT_LEN: int = 1000  # 10 s @ 100Hz
    BATCH_SIZE: int = 8
    SNR_DB: float = 20.0  # 设为 float('inf') 关闭加噪
    # 模型
    INPUT_BANDS: int = 64
    HIDDEN: int = 1024
    NUM_LAYERS: int = 1
    OUTPUT_SPACE: str = "OneHot"     # 'OneHot' or 'SRV'
    SRV_K: int = 10                  # for SRV
    SRV_N: int = 900                 # for SRV
    # 训练
    DEVICE: str = "cuda:0"
    LR: float = 1e-3
    STEPS_PER_EPOCH: int = 80
    EPOCHS: int = 20000
    PATIENCE: int = 200
    CKPT_DIR: Path = Path("./runs_pytorch")
    SEED: int = 42

cfg = Config()

# =========================
# 实用函数
# =========================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def db_to_noise_std(var: np.ndarray, snr_db: float):
    # var: (bands,)
    # snr = 10 * log10(sig/noise) => noise = sig / 10^(snr/10)
    return np.sqrt(var / (10.0 ** (snr_db / 10.0)))

# =========================
# 词表与输入加载
# =========================

class Lexicon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lexicon_txt = cfg.DATA_ROOT / "MALD-NEIGHBORS-1000.txt"
        self.speakers = (cfg.DATA_ROOT / "Speakers.txt").read_text().splitlines()
        gt_root = cfg.DATA_ROOT / "GAMMATONE_64_100"
        print("gt_root:", gt_root)
        self.words = [line.split("\t")[0] for line in self.lexicon_txt.read_text().splitlines()]
        self.n_words = len(self.words)
        self.word2idx = {w:i for i,w in enumerate(self.words)}
        # 读 gammatone：dict[(speaker, word)] -> (T, 64)
        self.inputs: Dict[Tuple[str,str], np.ndarray] = {}


        # 预热：计算每个频带方差，用于添加噪声
        gt_var = np.zeros(cfg.INPUT_BANDS, dtype=np.float64)
        count = 0
        for spk in self.speakers:
            dirname = spk if spk == 'MALD' else spk.title()
            folder = gt_root / dirname
            # 只加载必要的：这里直接全加载（方便），你也可以懒加载
            for w in self.words:
                p = folder / f"{w}_{spk.upper()}.pickle"  # 注意：文件名后缀是大写说话人
                with open(p, "rb") as f:
                    arr = pickle.load(f)  # (bands, T)
                arr = np.asarray(arr, dtype=np.float32).T  # (T, bands)
                self.inputs[(spk, w)] = arr
                gt_var += arr.var(axis=0)
                count += 1
        gt_var /= max(count, 1)
        self.input_var = gt_var.astype(np.float32)

    def build_targets(self):
        if cfg.OUTPUT_SPACE == "OneHot":
            self.out_dim = self.n_words
            self.embedding = np.eye(self.n_words, dtype=np.float32)
            self.kind = "onehot"
            self.srv_map = None
        elif cfg.OUTPUT_SPACE == "SRV":
            # 生成不太相似的 k-of-N 码
            rng = random.Random(42)
            patterns = []
            while len(patterns) < self.n_words:
                cand = set(rng.sample(range(cfg.SRV_N), cfg.SRV_K))
                # 简单唯一性约束（可加强）
                if any(len(cand.difference(p)) <= 2 for p in patterns):
                    continue
                patterns.append(cand)
            emb = np.zeros((self.n_words, cfg.SRV_N), dtype=np.float32)
            for i, idxs in enumerate(patterns):
                emb[i, list(idxs)] = 1.0
            self.out_dim = cfg.SRV_N
            self.embedding = emb
            self.kind = "srv"
            # 便于 min-pool 激活
            self.srv_map = [np.array(sorted(list(idxs)), dtype=np.int64) for idxs in patterns]
        else:
            raise ValueError("Unsupported OUTPUT_SPACE")

lex = Lexicon(cfg)
lex.build_targets()

# =========================
# 连续语流生成器（不重置隐状态）
# =========================

class WordToken:
    def __init__(self, t0:int, speaker:str, word:str, arr:np.ndarray):
        self.t0 = t0
        self.speaker = speaker
        self.word = word
        self.arr = arr  # (T, bands)
        self.t1 = t0 + arr.shape[0]

def gen_words_stream(lex: Lexicon, seed=0,
                     n_words=cfg.N_WORDS, n_phrases=cfg.N_PHRASES,
                     min_sil=cfg.MIN_SIL, max_sil=cfg.MAX_SIL):
    rng = random.Random(seed)
    # 分组：同一说话人串词
    items_by_spk: Dict[str, List[str]] = {}
    for spk in lex.speakers:
        items_by_spk[spk] = list(lex.words)
    speakers = list(items_by_spk.keys())
    spk = rng.choice(speakers)
    i_in_phrase = 0
    t = rng.randint(min_sil, max_sil) if n_words > 0 else 0
    while True:
        # 取一个词
        w = rng.choice(items_by_spk[spk])
        arr = lex.inputs[(spk, w)]  # (T,bands)
        # 嵌入静音（纯 0 帧）
        if n_words == 1 or (n_words and rng.randint(1, n_words) == 1):
            t += rng.randint(min_sil, max_sil)
        tok = WordToken(t, spk, w, arr)
        yield tok
        t = tok.t1
        # 短语说满后换说话人
        if n_phrases:
            i_in_phrase += 1
            if i_in_phrase >= n_phrases:
                spk = rng.choice(speakers)
                i_in_phrase = 0

def gen_segments(stream, seg_len=cfg.SEGMENT_LEN, out_dim=lex.out_dim,
                 embedding=lex.embedding, kind=lex.kind):
    """
    将连续 word 流切成固定长度段，输出：
    inputs:  (B=1, T, bands)
    targets: (B=1, T, out_dim) —— box 目标：词在其时段全 1
    备注：这里在段首/段尾自动分词（对齐逻辑与论文一致）
    """
    next_words: List[WordToken] = []
    t_start = 0
    while True:
        t_stop = t_start + seg_len
        words, q = [], next_words
        next_words = []
        for w in q:
            if w.t1 <= t_stop:
                words.append(w)
            else:
                # 被切开
                left_T = t_stop - w.t0
                words.append(WordToken(w.t0, w.speaker, w.word, w.arr[:left_T]))
                right = WordToken(t_stop, w.speaker, w.word, w.arr[left_T:])
                next_words.append(right)
        for w in stream:
            if w.t1 <= t_stop:
                words.append(w)
            elif w.t0 < t_stop:
                left_T = t_stop - w.t0
                words.append(WordToken(w.t0, w.speaker, w.word, w.arr[:left_T]))
                right = WordToken(t_stop, w.speaker, w.word, w.arr[left_T:])
                next_words.append(right)
                break
            else:
                next_words.append(w)
                break
        if not (words or next_words):
            return
        # 组 inputs / targets
        x = np.zeros((seg_len, cfg.INPUT_BANDS), dtype=np.float32)
        y = np.zeros((seg_len, out_dim), dtype=np.float32)
        for w in words:
            t0 = max(0, w.t0 - t_start)
            t1 = t0 + w.arr.shape[0]
            x[t0:t1] += w.arr
            # box target：整个词时段填同一个词的目标向量
            idx = lex.word2idx[w.word]
            vec = embedding[idx]  # (out_dim,)
            y[t0:t1, :out_dim] = vec
        yield x[None, ...], y[None, ...]  # (1,T,F), (1,T,D)
        t_start = t_stop

class BatchGenerator:
    """
    多条并行语流，维护每条流各自的段。用于“stateful”训练（每条流一个隐状态）。
    """
    def __init__(self, lex: Lexicon, batch_size=cfg.BATCH_SIZE, seed=cfg.SEED):
        self.lex = lex
        self.batch_size = batch_size
        rng = random.Random(seed)
        self.streams = [gen_words_stream(lex, seed=rng.getrandbits(31)) for _ in range(batch_size)]
        self.segments = [gen_segments(self.streams[i]) for i in range(batch_size)]

        # 噪声准备
        if math.isinf(cfg.SNR_DB):
            self.noise_std = None
        else:
            self.noise_std = db_to_noise_std(lex.input_var, cfg.SNR_DB).astype(np.float32)

    def next_batch(self):
        xs, ys = [], []
        alive = True
        for i in range(self.batch_size):
            try:
                x, y = next(self.segments[i])
            except StopIteration:
                alive = False
                break
            if self.noise_std is not None:
                noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
                x = x + noise
            xs.append(x); ys.append(y)
        if not alive:
            raise StopIteration
        x = np.concatenate(xs, axis=0)  # (B,T,F)
        y = np.concatenate(ys, axis=0)  # (B,T,D)
        return x, y

# =========================
# 模型
# =========================

class EarshotRNN(nn.Module):
    def __init__(self, input_bands, hidden, out_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_bands, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, state=None):
        # x: (B,T,F)
        y, state = self.rnn(x, state)  # y: (B,T,H)
        logits = self.out(y)           # (B,T,D)
        return logits, state

# =========================
# 训练循环（stateful + 早停）
# =========================

def train():
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    model = EarshotRNN(cfg.INPUT_BANDS, cfg.HIDDEN, lex.out_dim, cfg.NUM_LAYERS).to(device)
    # BCEWithLogits == sigmoid + BCE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=cfg.LR)

    # 目录
    run_name = f"LSTM{cfg.HIDDEN}_{cfg.OUTPUT_SPACE}_{cfg.SNR_DB}dB_{cfg.N_PHRASES}-{cfg.N_WORDS}_seg{cfg.SEGMENT_LEN}_bs{cfg.BATCH_SIZE}"
    out_dir = cfg.CKPT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 多流生成器（每条流一个 state）
    gen = BatchGenerator(lex, cfg.BATCH_SIZE, cfg.SEED)

    best_loss = float("inf")
    best_epoch = -1
    patience = cfg.PATIENCE
    state = None  # (h,c) 都是 (num_layers, B, H)

    # 简单日志
    log_path = out_dir / "log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,loss\n")

    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0.0
        # 每个 step 一个 10s 段（每条流各出一段）
        for step in tqdm(range(cfg.STEPS_PER_EPOCH), desc=f"Epoch {epoch}"):
            x_np, y_np = gen.next_batch()
            x = torch.from_numpy(x_np).to(device)   # (B,T,F)
            y = torch.from_numpy(y_np).to(device)   # (B,T,D)

            optimizer.zero_grad(set_to_none=True)
            logits, state = model(x, state)         # stateful：把 state 传回来继续用
            # 截断梯度流，避免跨 step 累积图
            if isinstance(state, tuple):
                state = (state[0].detach(), state[1].detach())
            else:
                state = state.detach()

            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= cfg.STEPS_PER_EPOCH
        print(f"[Epoch {epoch}] loss={epoch_loss:.6f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{epoch_loss:.6f}\n")

        # 早停 / 回滚
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_dir / "best.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_dir / f"epoch{epoch}.pt")
            patience = cfg.PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (loss={best_loss:.6f})")
                break

    print("Done. Best:", best_epoch, "Loss:", best_loss)

if __name__ == "__main__":
    train()
