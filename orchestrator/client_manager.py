import requests
import numpy as np
import time
import os
import json
from common.quantizacao import (
    quantize32to8,  dequantize8to32,
    quantize32to16, dequantize16to32,
)
from prometheus_client import Counter, Histogram, Gauge

CLIENT_ENDPOINTS = [
    "http://client-1:5000/fit",
    "http://client-2:5000/fit",
    "http://client-3:5000/fit",
]
MIN_TIMEOUT = 10
MAX_TIMEOUT = 300
CONNECT_TIMEOUT = 5
ALPHA = 0.125
BETA  = 0.25

# === Métricas Prometheus (orquestrador → clientes) ===
REQS = Counter("fl_requests_total", "Requisicoes a clientes", ["client","endpoint"])
HTTP_ERRS = Counter("fl_http_errors_total", "Erros HTTP por cliente", ["client","code"])
RETRIES = Counter("fl_retries_total", "Retries por cliente", ["client"])
TIMEOUTS = Counter("fl_timeouts_total", "Timeouts por cliente", ["client"])

RTT_MS = Histogram("fl_rtt_ms", "RTT (ms) por cliente", ["client"],
                   buckets=[5,10,20,30,50,75,100,150,200,300,500,1000,2000])
UPLOAD_B = Histogram("fl_upload_bytes", "Bytes enviados ao cliente", ["client"],
                     buckets=[1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7])
DOWNLOAD_B = Histogram("fl_download_bytes", "Bytes recebidos do cliente", ["client"],
                       buckets=[1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7])
THROUGHPUT_UP = Gauge("fl_throughput_up_mbps", "Upload MB/s", ["client"])
THROUGHPUT_DOWN = Gauge("fl_throughput_down_mbps", "Download MB/s", ["client"])

# --- helpers de (de)quantização unificados ---
def get_quant_mode():
    # "8" | "16" | "32"
    return str(os.getenv("QUANT_BITS", "8")).strip()

def build_payload(global_weights, quant_mode):
    """
    Retorna (payload_dict, overhead_quant_orchestrator_seconds)
    """
    if quant_mode == "32":
        # Sem compressão: manda float32 puro
        q_weights = [w.astype(np.float32).tolist() for w in global_weights]
        payload = {"quant": "32", "weights": q_weights, "scales": None, "zero_points": None}
        return payload, 0.0

    t0 = time.time()
    if quant_mode == "16":
        q, s, z = quantize32to16(global_weights)
        quant = "16"
    else:
        q, s, z = quantize32to8(global_weights)
        quant = "8"
    overhead = time.time() - t0

    payload = {"quant": quant, "weights": [w.tolist() for w in q], "scales": s, "zero_points": z}
    return payload, overhead

def dequantize_from_client(result):
    """
    Converte o que o cliente devolveu para float32.
    Usa result["quant"] (se não vier, assume mesmo modo enviado).
    Retorna (weights_float32, sample_count, over_quant, over_dequant)
    """
    quant = str(result.get("quant", "32"))
    over_q  = float(result.get("overhead_quant", 0.0))
    over_dq = float(result.get("overhead_dequant", 0.0))

    if quant == "32":
        w = [np.array(w, dtype=np.float32) for w in result["weights"]]
        return w, int(result["sample_count"]), over_q, over_dq
    elif quant == "16":
        wq = [np.array(w, dtype=np.int16) for w in result["weights"]]
        w  = dequantize16to32(wq, result["scales"], result["zero_points"])
        return w, int(result["sample_count"]), over_q, over_dq
    else:
        wq = [np.array(w, dtype=np.int8) for w in result["weights"]]
        w  = dequantize8to32(wq, result["scales"], result["zero_points"])
        return w, int(result["sample_count"]), over_q, over_dq

# --- função principal, compatível com 8/16/32 ---
def distribute_and_collect(global_weights, client_timing_stats):
    quant_mode = get_quant_mode()

    # payload de saída do orquestrador
    payload, overhead_quant_orchestrator = build_payload(global_weights, quant_mode)
    payload_json = json.dumps(payload).encode("utf-8")
    upload_payload_size = len(payload_json)

    client_updates = []
    metrics = {
        "orchestrator_quant_time": overhead_quant_orchestrator,
        "upload_payload_size_bytes": upload_payload_size,
        "client_metrics": [],
    }

    for i, endpoint in enumerate(CLIENT_ENDPOINTS):
        client_metric = {
            "client_id": i + 1,
            "download_payload_size_bytes": 0,
            "overhead_quant_client": 0.0,
            "overhead_dequant_client": 0.0,
            "rtt": 0.0,
            "status": "failure",
        }
        client_label = f"client-{i+1}"
        try:
            stats = client_timing_stats[endpoint]
            current_timeout = stats["avg_rtt"] + 4 * stats["dev_rtt"]
            current_timeout = max(MIN_TIMEOUT, min(current_timeout, MAX_TIMEOUT))

            print(f"Enviando modelo (quant={quant_mode}) para o cliente {i+1} ({endpoint})...")

            REQS.labels(client=client_label, endpoint="/fit").inc()
            t0 = time.time()
            response = requests.post(
                endpoint,
                json=payload,
                timeout=(CONNECT_TIMEOUT, current_timeout)
            )
            t1 = time.time()
            response.raise_for_status()

            # Métricas de rede
            sample_rtt = t1 - t0
            RTT_MS.labels(client=client_label).observe(sample_rtt * 1000.0)
            UPLOAD_B.labels(client=client_label).observe(upload_payload_size)
            down_len = len(response.content) if response.content else 0
            DOWNLOAD_B.labels(client=client_label).observe(down_len)
            rtt_s = max(sample_rtt, 1e-6)
            THROUGHPUT_UP.labels(client=client_label).set((upload_payload_size/1e6)/rtt_s)
            THROUGHPUT_DOWN.labels(client=client_label).set((down_len/1e6)/rtt_s)

            # Atualiza estatísticas adaptativas de RTT
            delta = abs(sample_rtt - stats["avg_rtt"])
            stats["dev_rtt"] = (1 - BETA) * stats["dev_rtt"] + BETA * delta
            stats["avg_rtt"] = (1 - ALPHA) * stats["avg_rtt"] + ALPHA * sample_rtt

            result = response.json()
            # fallback: se o cliente não devolveu "quant", assume o que enviamos
            if "quant" not in result:
                result["quant"] = quant_mode

            w_float32, sample_count, over_q, over_dq = dequantize_from_client(result)
            client_updates.append((w_float32, sample_count))

            client_metric["download_payload_size_bytes"] = down_len
            client_metric["overhead_quant_client"] = over_q
            client_metric["overhead_dequant_client"] = over_dq
            client_metric["rtt"] = sample_rtt
            client_metric["status"] = "success"
            metrics["client_metrics"].append(client_metric)

            print(f"Cliente {i+1} respondeu com sucesso (quant={result['quant']}).")

        except requests.exceptions.Timeout as e:
            TIMEOUTS.labels(client=client_label).inc()
            print(f"ERRO: Timeout contatando o cliente {i+1}. {e}")
            metrics["client_metrics"].append(client_metric)
        except requests.exceptions.RequestException as e:
            HTTP_ERRS.labels(client=client_label, code="conn_error").inc()
            print(f"ERRO: Não foi possível contatar o cliente {i+1}. {e}")
            metrics["client_metrics"].append(client_metric)

    return client_updates, client_timing_stats, metrics
