import unittest

from projeto import decisao


class TestDecisao(unittest.TestCase):
    def test_zumbi_no_limiar_aciona(self):
        self.assertEqual(decisao('zumbi', .80, .80), 'Z')

    def test_zumbi_abaixo_do_limiar_nao_aciona(self):
        self.assertEqual(decisao('zumbi', .799, .80), 'N')

    def test_normal_e_vazio_nao_acionam_mesmo_com_confianca_alta(self):
        for classe in ('normal', 'vazio'):
            with self.subTest(classe=classe):
                self.assertEqual(decisao(classe, 1.0, .80), 'N')

    def test_limiar_personalizado(self):
        self.assertEqual(decisao('zumbi', .85, .90), 'N')


if __name__ == '__main__':
    unittest.main()
