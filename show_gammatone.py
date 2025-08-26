from math import ceil, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle

root = Path("D:/00CODE00/dataset/Dataset/EARSHOT-gammatone+/GAMMATONE_64_100")


def single():
    # 选择一个文件（举例：Agnes 说话人的 “ABOUT”）

    speaker = "Agnes"  # 改成你实际的
    word = "ABOUT"  # 改成你实际的
    pkl = root / speaker / f"{word}_{speaker.upper()}.pickle"

    with open(pkl, "rb") as f:
        gt = pickle.load(f)  # numpy.ndarray, shape ≈ (n_bands=64, T)

    # 转 dB（避免值太小看不清）；你也可以直接画线性幅度
    eps = 1e-10
    gt_db = 20 * np.log10(gt + eps)

    # 时间轴（100 Hz → 每帧 0.01 秒）
    sr_time = 100.0
    T = gt.shape[1]
    t = np.arange(T) / sr_time

    print(gt_db.shape)#(64, 59)
    plt.figure(figsize=(9, 4))
    plt.imshow(gt_db, origin="lower", aspect="auto",
               extent=[t[0], t[-1], 0, gt.shape[0]])
    plt.xlabel("Time (s)")
    plt.ylabel("Gammatone band index (low→high)")
    plt.title(f"Gammatone (64 bands @100Hz): {word} [{speaker}]")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()


def erb_space(fmin=50.0, fmax=10000.0, n=64):
    # Glasberg & Moore ERB number,线性插值再反变换
    def hz2erb(f): return 21.4 * np.log10(4.37e-3 * f + 1.0)

    def erb2hz(e): return (10 ** (e / 21.4) - 1.0) / 4.37e-3

    erb = np.linspace(hz2erb(fmin), hz2erb(fmax), n)
    return erb2hz(erb)


def fre():
    speaker = "Agnes"
    word = "ABOUT"
    pkl = root / speaker / f"{word}_{speaker.upper()}.pickle"
    gt = pickle.loads(Path(pkl).read_bytes())

    # dB
    gt_db = 20 * np.log10(gt + 1e-10)

    # 时间轴
    sr_time = 100.0
    T = gt.shape[1]
    t = np.arange(T) / sr_time

    # 频率刻度
    cfs = erb_space(50.0, 10000.0, gt.shape[0])  # 64 个中心频率
    yticks = [0, 16, 32, 48, 63]  # 随意挑几个刻度
    ylabels = [f"{cfs[i]:.0f} Hz" for i in yticks]

    plt.figure(figsize=(9, 4))
    plt.imshow(gt_db, origin="lower", aspect="auto",
               extent=[t[0], t[-1], 0, gt.shape[0]])
    plt.yticks(yticks, ylabels)
    plt.xlabel("Time (s)")
    plt.ylabel("Center frequency")
    plt.title(f"Gammatone (64 bands @100Hz): {word} [{speaker}]")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()


def compare():
    speakers = ["Agnes", "Allison", "Bruce", "Junior", "Princess", "Samantha",
                "Tom", "Victoria", "Alex", "Ava", "Fred", "Kathy", "Ralph", "Susan", "Vicki", "MALD"]
    word = "ABOUT"  # 改成你要看的词

    fig, axes = plt.subplots(4, 4, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, spk in zip(axes, speakers):
        pkl = root / spk / f"{word}_{spk.upper()}.pickle"
        try:
            gt = pickle.loads(Path(pkl).read_bytes())
            gt_db = 20 * np.log10(gt + 1e-10)
            T = gt.shape[1];
            t = np.arange(T) / 100.0
            ax.imshow(gt_db, origin="lower", aspect="auto",
                      extent=[t[0], t[-1], 0, gt.shape[0]])
            ax.set_title(spk, fontsize=9)
        except FileNotFoundError:
            ax.set_title(f"{spk}\n(missing)", fontsize=9)
        ax.label_outer()

    fig.suptitle(f"Word: {word} — Gammatone 64×100Hz (dB)", y=0.995)
    fig.text(0.5, 0.04, "Time (s)", ha="center")
    fig.text(0.06, 0.5, "Band index (low→high)", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.show()


def load_gt(speaker: str, word: str):
    """读取某说话人的某单词的 gammatone pickle，返回 numpy 数组 (bands, T) 或 None。"""
    pkl = root / speaker / f"{word}_{speaker.upper()}.pickle"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


def list_words_for_speaker(speaker: str, stem_only=True):
    """列出某说话人可用的全部词。"""
    folder = root / speaker
    if not folder.exists():
        raise FileNotFoundError(f"Speaker folder not found: {folder}")
    words = []
    for p in folder.glob("*.pickle"):
        stem = p.stem  # e.g. ABOUT_AGNES
        w = stem[: stem.rfind("_")] if "_" in stem else stem
        words.append(w)
    words = sorted(set(words))
    return words


def plot_words_one_speaker(speaker: str, words, sr_time=100.0, db=True,
                           vmax_db=None, vmin_db=None,
                           cols=None, figsize=None, tight=True,
                           missing_as_blank=True, suptitle=True, save_path=None):
    """
    同一说话人，批量对比多个单词的 gammatone 频谱。
    - speaker: 说话人文件夹名（如 "Agnes" 或 "MALD"）
    - words:   词列表（如 ["ABOUT","ABOVE","ABLE", ...]）
    - db:      是否以 dB 显示
    - vmin_db/vmax_db: dB 范围，可设如 -80, -10；不设则自动
    - cols:    子图列数；不设则自动（≈ sqrt 布局）
    - figsize: 画布大小；不设则按子图数自适应
    - missing_as_blank: 缺文件时是否绘制空白框（True）或跳过（False）
    - save_path: 若给出路径则保存图片，否则 plt.show()
    """
    words = list(words)
    if cols is None:
        cols = int(ceil(sqrt(len(words))))
    rows = int(ceil(len(words) / cols))

    if figsize is None:
        figsize = (min(4 * cols, 16), min(3.2 * rows, 12))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    plotted = 0
    for i, word in enumerate(words):
        ax = axes[i]
        gt = load_gt(speaker, word)
        if gt is None:
            if missing_as_blank:
                ax.set_title(f"{word}\n(missing)", fontsize=9)
                ax.axis("off")
            else:
                ax.remove()
            continue

        arr = gt.astype(np.float32, copy=False)
        if db:
            arr = 20.0 * np.log10(arr + 1e-10)

        T = arr.shape[1]
        t = np.arange(T) / sr_time

        im = ax.imshow(arr, origin="lower", aspect="auto",
                       extent=[t[0], t[-1], 0, arr.shape[0]],
                       vmin=vmin_db if db and vmin_db is not None else None,
                       vmax=vmax_db if db and vmax_db is not None else None)
        ax.set_title(word, fontsize=10)
        ax.label_outer()
        plotted += 1

    # 清理多余子图
    for j in range(len(words), len(axes)):
        axes[j].axis("off")

    # 统一标签与色条
    fig.text(0.5, 0.04, "Time (s)", ha="center")
    fig.text(0.06, 0.5, "Band index (low→high)", va="center", rotation="vertical")
    cbar = fig.colorbar(im, ax=axes[:plotted], shrink=0.88)
    cbar.set_label("Amplitude (dB)" if db else "Amplitude")

    if suptitle:
        fig.suptitle(f"{speaker}: {len(words)} words — Gammatone 64×@100Hz", y=0.995)

    if tight:
        plt.tight_layout(rect=[0.06, 0.06, 0.995, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def compare_words():
    speaker = "Agnes"  # 或 "MALD"
    # 方式1：手动给词列表
    words = ["A", "ABOUT", "ABOVE", "ABLE", "AFTER", "ALWAYS", "AROUND", "ASK", "ACCENT", "BECAUSE", "BEFORE"]
    plot_words_one_speaker(speaker, words, db=True, vmin_db=-80, vmax_db=-10)

    # 方式2：自动列出这个说话人的前 N 个词看一眼
    all_words = list_words_for_speaker(speaker)
    plot_words_one_speaker(speaker, all_words[:12], db=True, vmin_db=-80, vmax_db=-10,
                           save_path=f"{speaker}_first12.png")


def _load_gt(speaker: str, word: str):
    p = root / speaker / f"{word}_{speaker.upper()}.pickle"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)  # (bands, T) numpy array

def _plot_cell(ax, arr, sr_time=100.0, db=True, vmin_db=None, vmax_db=None, title=None):
    if arr is None:
        ax.set_title(f"{title}\n(missing)" if title else "(missing)", fontsize=9)
        ax.axis("off")
        return None
    arr = arr.astype(np.float32, copy=False)
    if db:
        arr = 20.0 * np.log10(arr + 1e-10)
    T = arr.shape[1]; t = np.arange(T) / sr_time
    im = ax.imshow(
        arr, origin="lower", aspect="auto",
        extent=[t[0], t[-1], 0, arr.shape[0]],
        vmin=(vmin_db if (db and vmin_db is not None) else None),
        vmax=(vmax_db if (db and vmax_db is not None) else None)
    )
    if title:
        ax.set_title(title, fontsize=10)
    ax.label_outer()
    return im

def joint_compare(
    word_row="ABOUT",                     # 顶部：要比较的单词（跨说话人）
    speakers_row=("Agnes","Allison","Bruce","Junior","Princess","Samantha","Tom","Victoria",
                  "Alex","Ava","Fred","Kathy","Ralph","Susan","Vicki","MALD"),
    speaker_col="Agnes",                  # 底部：要比较的说话人（跨多个单词）
    words_col=("ABOUT","ABOVE","ABLE","AFTER","ALWAYS","AROUND","ASK","BECAUSE","BEFORE","AROUND"),
    db=True, vmin_db=-80, vmax_db=-10,   # dB 显示范围
    figsize=(14, 8), save_path=None
):
    """
    上：同一单词，不同说话人（1 × N）
    下：同一说话人，不同单词（1 × M）
    """
    n_top = len(speakers_row)
    n_bot = len(words_col)

    # 布局：2 行（上/下），列数为两者的最大值；不足的格子留空
    n_cols = max(n_top, n_bot)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize, sharex=True, sharey=True)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # 统一索引形状

    # —— 顶部：同词跨说话人 ——
    last_im = None
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_top:
            spk = speakers_row[i]
            gt = _load_gt(spk, word_row)
            title = spk
        else:
            gt, title = None, None
        last_im = _plot_cell(ax, gt, db=db, vmin_db=vmin_db, vmax_db=vmax_db, title=title)

    # —— 底部：同说话人跨词 ——
    for j in range(n_cols):
        ax = axes[1, j]
        if j < n_bot:
            w = words_col[j]
            gt = _load_gt(speaker_col, w)
            title = w
        else:
            gt, title = None, None
        last_im = _plot_cell(ax, gt, db=db, vmin_db=vmin_db, vmax_db=vmax_db, title=title)

    # 统一大标题与轴标
    axes[0, 0].set_ylabel("Band index (low→high)")
    axes[1, 0].set_ylabel("Band index (low→high)")
    fig.text(0.5, 0.04, "Time (s)", ha="center")

    # 色条
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Amplitude (dB)" if db else "Amplitude")

    # 行标题
    axes[0, 0].set_title(f"{speakers_row[0]}", fontsize=10)  # 左上已有标题，保持
    fig.suptitle(
        f"Top: word='{word_row}' across speakers | Bottom: speaker='{speaker_col}' across words",
        y=0.995
    )
    plt.tight_layout(rect=[0.04, 0.05, 0.995, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # joint_compare(
    #     word_row="ABOUT",
    #     speakers_row=("MALD", "Agnes", "Ava", "Bruce", "Fred", "Kathy", "Susan", "Tom"),
    #     speaker_col="MALD",
    #     words_col=("ABOUT", "ABOVE", "AFTER", "ALWAYS", "AROUND", "ASK", "BECAUSE", "BEFORE", "BETTER", "FATHER"),
    #     vmin_db=-80, vmax_db=-12,
    #     save_path="joint_compare_demo.png"
    # )
    single()
