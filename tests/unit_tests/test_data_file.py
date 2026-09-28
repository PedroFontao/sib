from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file, write_data_file


class TestDataFile(TestCase):

    def setUp(self):
        self.data_filename = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.data')

    def test_read_data_file(self):
        dataset = read_data_file(filename=self.data_filename, sep=",", label=True)

        self.assertEqual((699, 9), dataset.shape())
        self.assertEqual(True, dataset.has_label())
        self.assertEqual([0, 1], dataset.get_classes().tolist())

    def test_read_data_file_no_label(self):
        dataset = read_data_file(filename=self.data_filename, sep=",", label=False)

        self.assertEqual((699, 10), dataset.shape())
        self.assertEqual(False, dataset.has_label())

    def test_write_data_file(self):
        dataset = read_data_file(filename=self.data_filename, sep=",", label=True)
        write_data_file(filename='test.data', dataset=dataset, sep=",", label=True)
        dataset2 = read_data_file(filename='test.data', sep=",", label=True)

        self.assertEqual(dataset.shape(), dataset2.shape())
        self.assertEqual(dataset.has_label(), dataset2.has_label())
        self.assertEqual(dataset.get_classes().tolist(), dataset2.get_classes().tolist())
        os.remove('test.data')

    def test_write_data_file_no_label(self):
        dataset = read_data_file(filename=self.data_filename, sep=",", label=False)
        write_data_file(filename='test.data', dataset=dataset, sep=",", label=False)
        dataset2 = read_data_file(filename='test.data', sep=",", label=False)

        self.assertEqual(dataset.shape(), dataset2.shape())
        self.assertEqual(dataset.has_label(), dataset2.has_label())
        os.remove('test.data')
