import datetime
import os
import numpy as np
import time
import tensorflow as tf
import json
from common.model import create_simple_model
from orchestrator.client_manager import distribute_and_collect, CLIENT_ENDPOINTS
from orchestrator.aggregation import aggregate_weights

NUM_ROUNDS = 10

print("Carregando dados de teste do MNIST...")
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
print("Dados de teste carregados.")

def run_federated_training():
    print("--- Iniciando Treinamento Federado ---")
    start_total_time = time.time()
    
    global_model = create_simple_model()
    global_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    client_timing_stats = {endpoint: {"avg_rtt": 30.0, "dev_rtt": 5.0} for endpoint in CLIENT_ENDPOINTS}
    
    all_metrics_by_round = []

    for round_num in range(NUM_ROUNDS):
        print(f"\n--- RODADA {round_num + 1}/{NUM_ROUNDS} ---")
        
        start_round_time = time.time()
        
        global_weights = global_model.get_weights()

        # Distribuição e coleta dos pesos dos clientes
        client_updates, client_timing_stats, round_metrics = distribute_and_collect(global_weights, client_timing_stats)
        
        round_metrics['round'] = round_num + 1
        round_metrics['successful_clients'] = len(client_updates)
        round_metrics['round_duration_seconds'] = time.time() - start_round_time

        if not client_updates:
            print("Nenhum cliente respondeu. Pulando a rodada.")
            loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
            print(f"⚠️  AVALIAÇÃO GLOBAL (sem atualização) - Rodada {round_num + 1}: Perda = {loss:.4f}, Acurácia = {accuracy:.4f}")
            round_metrics['loss'] = loss
            round_metrics['accuracy'] = accuracy
            all_metrics_by_round.append(round_metrics)
            continue

        print("Agregando os pesos dos clientes...")
        new_weights = aggregate_weights(client_updates)

        global_model.set_weights(new_weights)
        print("Modelo global atualizado.")

        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ AVALIAÇÃO GLOBAL - Rodada {round_num + 1}: Perda = {loss:.4f}, Acurácia = {accuracy:.4f}")
        
        round_metrics['loss'] = loss
        round_metrics['accuracy'] = accuracy
        all_metrics_by_round.append(round_metrics)

        time.sleep(2)

    print("\n--- Treinamento Federado Concluído ---")
    end_total_time = time.time()
    total_training_duration = end_total_time - start_total_time
    print(f"Tempo total de treinamento: {total_training_duration:.2f} segundos")
    
    final_results = {
        'total_training_duration_seconds': total_training_duration,
        'metrics_by_round': all_metrics_by_round
    }

    run_id = os.environ.get("POD_NAME", "run") + "-" + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    os.makedirs('/app/metrics', exist_ok=True)
    out_path = f'/app/metrics/federated_training_metrics-8-atrasos-{run_id}.json'
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=4)
        print("Métricas de treinamento federado salvas em 'metrics/federated_training_metrics-.json'.")

if __name__ == '__main__':
    print("Orquestrador esperando 10 segundos para os clientes iniciarem...")
    time.sleep(10)
    run_federated_training()