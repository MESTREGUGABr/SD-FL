import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "font.size": 12,
})

def load_metrics(paths_or_dirs):
    files = []
    for p in paths_or_dirs:
        if os.path.isdir(p):
            files += [os.path.join(p,f) for f in os.listdir(p) if f.endswith(".json")]
        else:
            if p.endswith(".json"): files.append(p)

    rounds_rows, clients_rows = [], []
    for fp in files:
        with open(fp) as f: d = json.load(f)
        for r in d['metrics_by_round']:
            rtts = [cm['rtt'] for cm in r['client_metrics']]
            rounds_rows.append({
                'round': r['round'],
                'round_duration': r['round_duration_seconds'],
                'loss': r['loss'],
                'accuracy': r['accuracy'],
                'upload_bytes': r.get('upload_payload_size_bytes', np.nan),
                'rtt_max': np.max(rtts),
            })
            for cm in r['client_metrics']:
                clients_rows.append({
                    'round': r['round'],
                    'client_id': cm['client_id'],
                    'rtt': cm['rtt'],
                    'download_bytes': cm.get('download_payload_size_bytes', np.nan),
                })
    return pd.DataFrame(rounds_rows), pd.DataFrame(clients_rows)

def plot_all(paths_or_dirs, outdir="plots_clean"):
    os.makedirs(outdir, exist_ok=True)
    rf, cf = load_metrics(paths_or_dirs)

    # remove duplicatas de round
    rf = rf.groupby('round', as_index=False).mean(numeric_only=True)

    # 1) Accuracy vs Round
    plt.figure(figsize=(7,4))
    plt.plot(rf['round'], rf['accuracy'], linewidth=2, marker='o')
    plt.title("Acurácia por Rodada"); plt.xlabel("Rodada"); plt.ylabel("Acurácia")
    plt.xticks(rf['round']); plt.grid(True, ls=":")
    plt.tight_layout(); plt.savefig(f"{outdir}/01_accuracy.png"); plt.close()

    # 2) RTT médio por cliente
    agg = cf.groupby('client_id')['rtt'].agg(['mean','std'])
    plt.figure(figsize=(6,4))
    plt.bar(agg.index.astype(str), agg['mean'], yerr=agg['std'], capsize=5)
    plt.title("RTT médio por Cliente (±DP)"); plt.xlabel("Cliente"); plt.ylabel("RTT (s)")
    plt.tight_layout(); plt.savefig(f"{outdir}/02_rtt_by_client.png"); plt.close()

    # 3) Tempo médio de cada cliente (via RTT)
    # aqui basicamente é o mesmo do gráfico anterior, só mudamos o rótulo
    plt.figure(figsize=(6,4))
    plt.bar(agg.index.astype(str), agg['mean'], capsize=5, color="orange")
    plt.title("Tempo médio por Cliente"); plt.xlabel("Cliente"); plt.ylabel("Tempo médio (s)")
    plt.tight_layout(); plt.savefig(f"{outdir}/03_client_time.png"); plt.close()

    # 4) Round Duration vs RTT máximo
    plt.figure(figsize=(6.5,4.5))
    plt.scatter(rf['rtt_max'], rf['round_duration'], s=50)
    plt.title("Duração da Rodada vs RTT Máximo")
    plt.xlabel("RTT máximo (s)"); plt.ylabel("Duração da Rodada (s)")
    plt.tight_layout(); plt.savefig(f"{outdir}/04_round_vs_rttmax.png"); plt.close()

    # 5) Payload por rodada
    mean_download = cf.groupby('round')['download_bytes'].mean()
    plt.figure(figsize=(7,4.5))
    plt.plot(rf['round'], rf['upload_bytes'], marker='o', label='Upload (orq.→clientes)')
    plt.plot(mean_download.index, mean_download.values, marker='s', label='Download médio (cliente)')
    plt.title("Tamanho de Payload por Rodada")
    plt.xlabel("Rodada"); plt.ylabel("Bytes"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/05_payload.png"); plt.close()


if __name__ == "__main__":
    plot_all(["metricas"])
