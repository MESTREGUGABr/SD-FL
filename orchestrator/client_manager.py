import requests
import numpy as np
import time
import os
import json
from common.quantizacao import quantize32to8, dequantize8to32

# Constantes e configurações
CLIENT_ENDPOINTS = [
    "http://client-1:5000/fit",
    "http://client-2:5000/fit",
    "http://client-3:5000/fit",
]
MIN_TIMEOUT = 10
MAX_TIMEOUT = 180
ALPHA = 0.125
BETA = 0.25

def distribute_and_collect(global_weights, client_timing_stats):
    #Quantiza os pesos globais para 8 bits
    t_quant_orchestrator = time.time()
    q_weights, scales, zero_points = quantize32to8(global_weights)
    overhead_quant_orchestrator = time.time() - t_quant_orchestrator

    payload = {
        "weights": [w.tolist() for w in q_weights],
        "scales": scales,
        "zero_points": zero_points
    }
    
    upload_payload_size = len(json.dumps(payload).encode('utf-8'))

    client_updates = []
    metrics = {
        'orchestrator_quant_time': overhead_quant_orchestrator,
        'upload_payload_size_bytes': upload_payload_size,
        'client_metrics': []
    }

    for i, endpoint in enumerate(CLIENT_ENDPOINTS):
        client_metric = {
            'client_id': i + 1,
            'download_payload_size_bytes': 0,
            'overhead_quant_client': 0,
            'overhead_dequant_client': 0,
            'rtt': 0,
            'status': 'failure'
        }
        try:
            stats = client_timing_stats[endpoint]
            current_timeout = stats["avg_rtt"] + 4 * stats["dev_rtt"]
            current_timeout = max(MIN_TIMEOUT, min(current_timeout, MAX_TIMEOUT))

            print(f"Enviando modelo para o cliente {i+1} ({endpoint})...")

            start_time = time.time()
            response = requests.post(endpoint, json=payload, timeout=current_timeout)
            end_time = time.time()
            
            response.raise_for_status()

            # Atualiza as estatísticas de tempo
            sample_rtt = end_time - start_time
            delta = abs(sample_rtt - stats["avg_rtt"])
            stats["dev_rtt"] = (1 - BETA) * stats["dev_rtt"] + BETA * delta
            stats["avg_rtt"] = (1 - ALPHA) * stats["avg_rtt"] + ALPHA * sample_rtt

            result = response.json()
            
            # Dequantiza os pesos recebidos do cliente
            client_weights_q = [np.array(w, dtype=np.int8) for w in result['weights']]
            client_weights = dequantize8to32(client_weights_q, result['scales'], result['zero_points'])
            sample_count = result['sample_count']

            client_updates.append((client_weights, sample_count))
            
            client_metric['download_payload_size_bytes'] = len(response.content)
            client_metric['overhead_quant_client'] = result['overhead_quant']
            client_metric['overhead_dequant_client'] = result['overhead_dequant']
            client_metric['rtt'] = sample_rtt
            client_metric['status'] = 'success'
            metrics['client_metrics'].append(client_metric)

            print(f"Cliente {i+1} respondeu com sucesso.")

        except requests.exceptions.RequestException as e:
            print(f"ERRO: Não foi possível contatar o cliente {i+1}. {e}")
            metrics['client_metrics'].append(client_metric)
            
    return client_updates, client_timing_stats, metrics