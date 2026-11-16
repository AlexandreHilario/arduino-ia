"""Treina uma CNN do zero, sem pesos ou modelos pre-treinados."""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CLASSES = ("zumbi", "normal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=BASE / "dados")
    parser.add_argument("--saida", type=Path, default=BASE / "modelo_do_zero")
    parser.add_argument("--epocas", type=int, default=50)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--incluir-vazio", action="store_true",
                        help="Treinar tambem com fotos em dados/vazio")
    args = parser.parse_args()
    if args.epocas < 1:
        parser.error("Use pelo menos uma epoca.")
    classes = CLASSES + (("vazio",) if args.incluir_vazio else ())

    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageOps

    tf.keras.utils.set_random_seed(args.semente)
    rng = np.random.default_rng(args.semente)
    treino, validacao = [], []
    manifesto = {"treino": [], "validacao": []}
    # Divisao por classe antes de qualquer aumento artificial das imagens.
    for indice, classe in enumerate(classes):
        fotos = sorted(p for p in (args.dados / classe).glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(fotos) < 20:
            parser.error(f"Separe pelo menos 20 fotos em {args.dados / classe}; recomendado: 200 ou mais.")
        rng.shuffle(fotos)
        quantidade_validacao = max(1, round(len(fotos) * .15))
        for posicao, foto in enumerate(fotos):
            with Image.open(foto) as imagem:
                imagem = ImageOps.fit(imagem.convert("RGB"), (96, 96), Image.Resampling.LANCZOS)
                tensor = np.asarray(imagem, dtype=np.float32) / 127.5 - 1.0
            nome = "validacao" if posicao < quantidade_validacao else "treino"
            (validacao if nome == "validacao" else treino).append((tensor, indice))
            manifesto[nome].append(str(foto.resolve()))

    def arrays(amostras):
        return np.stack([x for x, _ in amostras]), np.array([y for _, y in amostras])

    x_treino, y_treino = arrays(treino)
    x_val, y_val = arrays(validacao)
    # Todas as camadas aprendem a partir de inicializacao aleatoria.
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(96, 96, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(.05, fill_mode="reflect"),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.summary()
    historico = modelo.fit(
        x_treino, y_treino, validation_data=(x_val, y_val),
        epochs=args.epocas, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)],
    )
    perda, acuracia = modelo.evaluate(x_val, y_val, verbose=0)
    previstas = np.argmax(modelo.predict(x_val, verbose=0), axis=1)
    matriz = tf.math.confusion_matrix(y_val, previstas, num_classes=len(classes)).numpy().tolist()
    args.saida.mkdir(parents=True, exist_ok=True)
    modelo.save(args.saida / "modelo.keras")
    # Exportacao explicita de inferencia: aumento de dados e dropout desligados.
    @tf.function(input_signature=[tf.TensorSpec([1, 96, 96, 3], tf.float32)])
    def inferir(imagens):
        return modelo(imagens, training=False)

    concreto = inferir.get_concrete_function()
    conversor = tf.lite.TFLiteConverter.from_concrete_functions([concreto], modelo)
    conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    convertido = conversor.convert()
    # Confere a exportacao contra a rede original antes de gravar o TFLite.
    interpretador = tf.lite.Interpreter(model_content=convertido)
    interpretador.allocate_tensors()
    entrada = interpretador.get_input_details()[0]
    saida = interpretador.get_output_details()[0]
    amostra = x_val[:1]
    interpretador.set_tensor(entrada["index"], amostra)
    interpretador.invoke()
    np.testing.assert_allclose(interpretador.get_tensor(saida["index"]),
                               modelo(amostra, training=False).numpy(), atol=1e-4, rtol=1e-3)
    (args.saida / "model_unquant.tflite").write_bytes(convertido)
    (args.saida / "labels.txt").write_text(
        "".join(f"{i} {classe}\n" for i, classe in enumerate(classes)), encoding="utf-8")
    relatorio = {"origem": "CNN com pesos aleatorios, sem pre-treinamento",
                 "semente": args.semente, "classes": classes,
                 "divisao": manifesto, "historico": historico.history,
                 "validacao": {"perda": float(perda), "acuracia": float(acuracia),
                               "matriz_confusao_real_por_prevista": matriz}}
    (args.saida / "treinamento.json").write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    print(f"Modelo e relatorio salvos em {args.saida}")


if __name__ == "__main__":
    main()
