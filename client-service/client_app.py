import time
from flask import Flask, request, jsonify, Response
import tensorflow as tf
import numpy as np
from common.model import create_simple_model
from common.quantizacao import quantize32to8, dequantize8to32, dequantize16to32, quantize32to16
import os
import traceback

app = Flask(__name__)

print("Carregando dados do MNIST...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

client_id = int(os.environ.get('CLIENT_ID', '0'))
start_index = client_id * 1000
end_index = start_index + 1000
local_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train[start_index:end_index], y_train[start_index:end_index])
).batch(32)
sample_count = end_index - start_index

print(f"Cliente {client_id} iniciado com dados do índice {start_index} ao {end_index}.")


def train_local_model(weights):
    model = create_simple_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.set_weights(weights)
    
    model.fit(local_dataset, epochs=1, verbose=0)
    
    new_weights = model.get_weights()
    return new_weights


@app.route('/fit', methods=['POST'])
def fit():
    try:
        payload = request.json
        quant = str(payload.get('quant', '8'))
        t0 = time.time()
        if quant == '32':
            weights = [np.array(w, dtype=np.float32) for w in payload['weights']]
            dequant_time = 0.0
        elif quant == '16':
            weights_q = [np.array(w, dtype=np.int16) for w in payload['weights']]
            scales = payload['scales']; zero_points = payload['zero_points']
            weights = dequantize16to32(weights_q, scales, zero_points)
            dequant_time = time.time() - t0
        else:  
            weights_q = [np.array(w, dtype=np.int8) for w in payload['weights']]
            scales = payload['scales']; zero_points = payload['zero_points']
            weights = dequantize8to32(weights_q, scales, zero_points)
            dequant_time = time.time() - t0
        
        updated_weights = train_local_model(weights)

        t1 = time.time()
        if quant == '32':
            q_new = [w.astype(np.float32) for w in updated_weights]
            new_scales, new_zps = None, None
        elif quant == '16':
            q_new, new_scales, new_zps = quantize32to16(updated_weights)
        else:  
            q_new, new_scales, new_zps = quantize32to8(updated_weights)
        quant_time = time.time() - t1 if quant != '32' else 0.0
        
        return jsonify({
            "weights": [w.tolist() for w in q_new],
            "scales": new_scales,
            "zero_points": new_zps,
            "sample_count": sample_count,
            "overhead_quant": quant_time,
            "overhead_dequant": dequant_time
        })

    except Exception as e:
        print(f"ERRO CRÍTICO no cliente {client_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)