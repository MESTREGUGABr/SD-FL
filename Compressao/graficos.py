import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot_federated_metrics(directories):
    """
    Reads federated learning metrics from multiple directories, aggregates the data,
    and generates comparative plots.

    Args:
        directories (list): A list of directory names, e.g., ['8b', '16b', '32b'].
    """
    all_data = {}

    for d in directories:
        all_metrics_list = []
        all_total_durations = []
        for filename in os.listdir(d):
            if filename.endswith('.json'):
                file_path = os.path.join(d, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_metrics_list.extend(data['metrics_by_round'])
                    all_total_durations.append(data['total_training_duration_seconds'])

        df = pd.DataFrame(all_metrics_list)
        
        # Select only the numeric columns before calculating the mean
        numeric_cols = ['loss', 'accuracy', 'round_duration_seconds', 'orchestrator_quant_time']
        df_numeric = df[numeric_cols]
        
        all_data[d] = {
            'metrics_per_round': df_numeric.groupby(df['round']).mean(),
            'total_duration_mean': np.mean(all_total_durations)
        }

    # --- Generate the plots ---

    # Plot 1: Total Training Duration by Quantization Level
    total_durations = [all_data[d]['total_duration_mean'] for d in directories]
    quantization_labels = [d.replace('b', '-bit') for d in directories]

    plt.figure(figsize=(8, 6))
    plt.bar(quantization_labels, total_durations, color='orange')
    plt.title('Total Training Duration by Quantization Level')
    plt.xlabel('Quantization Bits')
    plt.ylabel('Total Training Duration (s)')
    for i, v in enumerate(total_durations):
        plt.text(i, v + 1, f"{v:.1f}", ha='center', va='bottom')
    plt.savefig('Total_Training_Duration.png')
    plt.close()

    # Plot 2: Accuracy per Round
    plt.figure(figsize=(10, 6))
    for d in directories:
        plt.plot(all_data[d]['metrics_per_round'].index, all_data[d]['metrics_per_round']['accuracy'], label=d.replace('b', '-bit'))
    plt.title('Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('Accuracy_per_Round.png')
    plt.close()

    # Plot 3: Round Duration per Round
    plt.figure(figsize=(10, 6))
    for d in directories:
        plt.plot(all_data[d]['metrics_per_round'].index, all_data[d]['metrics_per_round']['round_duration_seconds'], label=d.replace('b', '-bit'))
    plt.title('Round Duration per Round')
    plt.xlabel('Round')
    plt.ylabel('Round Duration (s)')
    plt.grid(True)
    plt.legend()
    plt.savefig('Round_Duration_per_Round.png')
    plt.close()

    # Plot 4: Orchestrator Quantization Overhead per Round
    plt.figure(figsize=(10, 6))
    for d in directories:
        plt.plot(all_data[d]['metrics_per_round'].index, all_data[d]['metrics_per_round']['orchestrator_quant_time'], label=d.replace('b', '-bit'))
    plt.title('Orchestrator Quantization Overhead per Round')
    plt.xlabel('Round')
    plt.ylabel('Quantization Overhead (s)')
    plt.grid(True)
    plt.legend()
    plt.savefig('Orchestrator_Quantization_Overhead.png')
    plt.close()

    # # Resilience data and plot (manual data as it's not in the provided JSONs)
    # resilience_data = {
    #     'Scenario': ['no failures', 'Single Node Failure', 'Multiple Node Failure'],
    #     'Score': [1.00, 0.80, 0.65]
    # }
    # df_resilience = pd.DataFrame(resilience_data)

    # plt.figure(figsize=(8, 6))
    # plt.bar(df_resilience['Scenario'], df_resilience['Score'], color='orange')
    # plt.title('Resilience Score by Scenario')
    # plt.xlabel('Scenario')
    # plt.ylabel('Resilience Score')
    # for i, v in enumerate(df_resilience['Score']):
    #     plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    # plt.ylim(0, 1.1)
    # plt.savefig('Resilience_Score.png')
    # plt.close()

# Main call to run the analysis and plotting
directories = ['8b', '16b', '32b']
analyze_and_plot_federated_metrics(directories)