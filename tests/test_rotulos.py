import tempfile
import unittest
from pathlib import Path
from projeto import BASE, PASTA_MODELO, carregar_classes


class TestRotulos(unittest.TestCase):
    def ler(self, texto):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'labels.txt'
            arquivo.write_text(texto, encoding='utf-8')
            return carregar_classes(arquivo)

    def test_exportacao_incluida(self):
        self.assertTrue((PASTA_MODELO / 'model_unquant.tflite').is_file())
        self.assertEqual(carregar_classes(PASTA_MODELO / 'labels.txt'), ['zumbi', 'normal'])

    def test_preserva_ordem_das_saidas(self):
        self.assertEqual(self.ler('0 Pessoa\n1 Zumbi\n'), ['normal', 'zumbi'])

    def test_tres_classes(self):
        self.assertEqual(self.ler('0 Vazio\n1 Normal\n2 Zumbi'), ['vazio', 'normal', 'zumbi'])

    def test_rejeita_rotulos_incompativeis(self):
        for texto in ('0 Pessoa\n1 Vazio', '0 Zumbi\n1 Normal\n2 Pessoa', '0 Zumbi\n1 Class 2'):
            with self.subTest(texto=texto), self.assertRaises(ValueError):
                self.ler(texto)
