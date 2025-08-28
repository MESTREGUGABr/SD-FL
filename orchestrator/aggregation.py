# orchestrator/aggregation.py

import numpy as np

def aggregate_weights(client_updates):
    if not client_updates:
        return []

    # Calcular o número total de amostras de todos os clientes que responderam
    total_samples = sum(sample_count for _, sample_count in client_updates)
    
    # Inicializar os pesos agregados com zeros
    first_client_weights = client_updates[0][0]
    new_weights = [np.zeros_like(w) for w in first_client_weights]

    # Agregação ponderada
    for client_weights, sample_count in client_updates:
        weight_contribution = sample_count / total_samples
        for i in range(len(new_weights)):
            new_weights[i] += client_weights[i] * weight_contribution

    return new_weights