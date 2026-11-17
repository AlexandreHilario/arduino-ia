"""Captura, treinamento e reconhecimento do boneco de zumbi."""
import argparse
from pathlib import Path
import time
import uuid

BASE = Path(__file__).resolve().parent
CLASSES = ("zumbi", "normal", "vazio")
PASTA_MODELO = BASE / "converted_tflite"


def camera(indice):
    import cv2
    origem = int(indice) if str(indice).isdigit() else indice
    if isinstance(origem, str) and origem.startswith("/dev/video"):
        cap = cv2.VideoCapture(origem, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(origem)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Nao foi possivel abrir a camera {indice}. Confira a origem e se o scrcpy esta ativo.")
    print(f"Camera selecionada: {indice}")
    return cap


def capturar(args):
    import cv2
    pasta = BASE / "dados" / args.classe
    pasta.mkdir(parents=True, exist_ok=True)
    cap = camera(args.camera)
    quantidade = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler a camera.")
            preview = frame.copy()
            cv2.putText(preview, f"{args.classe}: {quantidade} | ESPACO: foto | Q: sair",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow("Captura", preview)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == 32:
                destino = pasta / f"{uuid.uuid4().hex}.jpg"
                if not cv2.imwrite(str(destino), frame):
                    raise RuntimeError(f"Nao foi possivel salvar {destino}")
                quantidade += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def carregar_classes(rotulos):
    """Preserva a ordem da exportacao e normaliza os nomes conhecidos."""
    classes = []
    for linha in Path(rotulos).read_text(encoding="utf-8-sig").splitlines():
        partes = linha.strip().split(maxsplit=1)
        if not partes:
            continue
        nome = partes[1] if partes[0].isdigit() and len(partes) == 2 else linha.strip()
        nome = nome.casefold()
        classes.append("normal" if nome == "pessoa" else nome)
    if (len(classes) < 2 or len(set(classes)) != len(classes)
            or "zumbi" not in classes or not set(classes).issubset(CLASSES)):
        raise ValueError("labels.txt deve conter Zumbi e Normal/Pessoa e/ou Vazio, sem duplicatas.")
    return classes


class Modelo:
    """Modelo float32 de imagens exportado pelo Teachable Machine."""

    def __init__(self, caminho, rotulos):
        import numpy as np
        import tensorflow as tf
        self.np = np
        self.classes = carregar_classes(rotulos)
        self.rede = tf.lite.Interpreter(model_path=str(caminho), num_threads=2)
        self.rede.allocate_tensors()
        entradas, saidas = self.rede.get_input_details(), self.rede.get_output_details()
        if len(entradas) != 1 or len(saidas) != 1:
            raise ValueError("Exporte um modelo de classificacao de imagens.")
        self.entrada, self.saida = entradas[0], saidas[0]
        forma = self.entrada["shape"]
        if len(forma) != 4 or forma[0] != 1 or forma[3] != 3:
            raise ValueError("Entrada esperada: uma imagem RGB.")
        if self.entrada["dtype"] != np.float32 or self.saida["dtype"] != np.float32:
            raise ValueError("Exporte TensorFlow Lite no formato Floating point (float32).")
        if tuple(self.saida["shape"]) != (1, len(self.classes)):
            raise ValueError("O numero de saidas do modelo deve corresponder ao labels.txt.")
        self.tamanho = (int(forma[2]), int(forma[1]))

    def prever(self, imagem):
        from PIL import Image, ImageOps
        np = self.np
        imagem = ImageOps.fit(imagem.convert("RGB"), self.tamanho, Image.Resampling.LANCZOS)
        tensor = (np.asarray(imagem, dtype=np.float32) / 127.5 - 1.0)[None, ...]
        self.rede.set_tensor(self.entrada["index"], tensor)
        self.rede.invoke()
        valores = self.rede.get_tensor(self.saida["index"])[0]
        if not np.all(np.isfinite(valores)) or np.any(valores < 0) or np.any(valores > 1):
            raise ValueError("Saida do modelo nao contem probabilidades validas.")
        indice = int(np.argmax(valores))
        return self.classes[indice], float(valores[indice])


def decisao(classe, confianca, limiar):
    return "Z" if classe == "zumbi" and confianca >= limiar else "N"


def avaliar(args):
    import json
    from PIL import Image
    modelo = Modelo(args.modelo, args.rotulos)
    classes = modelo.classes
    matriz = [[0] * len(classes) for _ in classes]
    matriz_decisao = [[0] * 2 for _ in range(2)]
    totais = {}
    for i, classe in enumerate(classes):
        fotos = sorted(p for p in (args.dados / classe).glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(fotos) < 20:
            raise ValueError(f"Separe pelo menos 20 imagens novas em {args.dados / classe}.")
        totais[classe] = len(fotos)
        for foto in fotos:
            with Image.open(foto) as imagem:
                prevista, confianca = modelo.prever(imagem)
            matriz[i][classes.index(prevista)] += 1
            matriz_decisao[int(classe == "zumbi")][int(decisao(prevista, confianca, args.confianca) == "Z")] += 1
    resultado = {"classes": classes, "totais": totais,
                 "acuracia": sum(matriz[i][i] for i in range(len(classes))) / sum(totais.values()),
                 "matriz_confusao_real_por_prevista": matriz,
                 "limiar": args.confianca,
                 "matriz_decisao_real_por_prevista_N_Z": matriz_decisao}
    args.relatorio.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(json.dumps(resultado, indent=2))


def executar(args):
    import cv2
    import serial
    from PIL import Image
    modelo = Modelo(args.modelo, args.rotulos)
    cap = camera(args.camera)
    conexao = None
    buffer = b""
    texto = "Aguardando ANALISAR do Arduino..."
    try:
        if not args.sem_arduino:
            conexao = serial.Serial(args.porta, 9600, timeout=0, write_timeout=1)
            time.sleep(2)
            conexao.reset_input_buffer()
        while True:
            analisar = args.sem_arduino
            if conexao:
                buffer += conexao.read(conexao.in_waiting)
                while b"\n" in buffer:
                    linha, buffer = buffer.split(b"\n", 1)
                    if linha.strip() == b"ANALISAR":
                        analisar = True
                    elif linha.strip() == b"LONGE":
                        analisar = False
                        texto = "Aguardando aproximacao (< 200 cm)..."
                buffer = buffer[-128:]
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler a camera.")
            if analisar:
                imagem = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                classe, confianca = modelo.prever(imagem)
                resposta = decisao(classe, confianca, args.confianca)
                if conexao:
                    conexao.write((resposta + "\n").encode("ascii"))
                texto = f"{classe}: {confianca:.0%} | resposta: {resposta}"
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.imshow("Torre anti-zumbi - Q para sair", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if conexao:
            try:
                conexao.write(b"N\n")
            finally:
                conexao.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    captura = comandos.add_parser("capturar", help="Tirar fotos com ESPACO")
    captura.add_argument("classe", choices=CLASSES)
    captura.add_argument("--camera", default="0", help="Indice ou caminho, como /dev/video10")
    captura.set_defaults(func=capturar)
    executar_parser = comandos.add_parser("executar", help="Responder ao Arduino pela camera")
    executar_parser.add_argument("--porta", default="COM3")
    executar_parser.add_argument("--camera", default="0", help="Indice ou caminho, como /dev/video10")
    executar_parser.add_argument("--sem-arduino", action="store_true")
    executar_parser.set_defaults(func=executar)
    avaliacao = comandos.add_parser("avaliar", help="Avaliar imagens novas, separadas do treino")
    avaliacao.add_argument("--dados", type=Path, default=BASE / "testes")
    avaliacao.add_argument("--relatorio", type=Path, default=BASE / "avaliacao.json")
    avaliacao.set_defaults(func=avaliar)
    for comando in (executar_parser, avaliacao):
        comando.add_argument("--modelo", type=Path, default=PASTA_MODELO / "model_unquant.tflite")
        comando.add_argument("--rotulos", type=Path, default=PASTA_MODELO / "labels.txt")
        comando.add_argument("--confianca", type=float, default=.80)
    args = parser.parse_args()
    if args.comando in ("executar", "avaliar"):
        if not 0 < args.confianca <= 1:
            parser.error("Use confianca maior que 0 e ate 1.")
    try:
        args.func(args)
    except (RuntimeError, ValueError, OSError, ImportError) as erro:
        parser.exit(1, f"Erro: {erro}\n")
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main()
