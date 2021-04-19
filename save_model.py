import tensorflow as tf
import os
import argparse
import json
import pickle
import dnnlib
import dnnlib.tflib as tflib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', help='Network pickle filename', dest='network', required=True)
    parser.add_argument('--destination', help='Output h5 filename', dest='destination', required=True)
    
    args = parser.parse_args()
    
    tflib.init_tf()
    with open(args.network, 'rb') as f:
        _, _, Gs = pickle.load(f)
    print(f"loaded model: {args.network}")

    tf.keras.models.save_model(
        Gs,
        args.destination,
        overwrite=True,
        include_optimizer=True,
        save_format="h5",
        signatures=None
    )
    
    print(f"saved model to {args.destination}")

    
if __name__ == "__main__":
    main()
