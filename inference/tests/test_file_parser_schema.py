"""Check that schema fallbacks reach the tool response, including cache hits."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_tools import file_parser


class TestFileParserSchema(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.parser = file_parser.SingleFileParser({'path': str(self.directory / 'cache')})

    def test_large_csv_schema_is_returned_and_cached(self):
        path = self.directory / 'cities.csv'
        path.write_text('city,temperature\n' + '北京,21\n' * 15000, encoding='utf-8')

        response = self.parser.call({'url': str(path)})
        schema = json.loads(response)
        self.assertEqual(schema['columns'], ['city', 'temperature'])
        self.assertEqual(schema['sample_data']['city'], ['北京'] * 3)
        with patch.object(self.parser, '_process_new_file', side_effect=AssertionError('cache miss')):
            self.assertEqual(self.parser.call({'url': str(path)}), response)

    def test_large_xml_schema_is_returned_and_cached(self):
        path = self.directory / 'cities.xml'
        content = '<cities>' + '<city>北京</city>' * 15000 + '</cities>'
        self.assertGreater(file_parser.count_tokens(content), file_parser.DEFAULT_MAX_INPUT_TOKENS)
        path.write_text(content, encoding='utf-8')

        response = self.parser.call({'url': str(path)})
        self.assertIn('## Root: cities', response)
        self.assertIn('**city**', response)
        with patch.object(self.parser, '_process_new_file', side_effect=AssertionError('cache miss')):
            self.assertEqual(self.parser.call({'url': str(path)}), response)

    def test_small_text_and_table_still_return_content(self):
        path = self.directory / 'note.txt'
        path.write_text('A short note', encoding='utf-8')
        self.assertEqual(self.parser.call({'url': str(path)}), 'A short note')
        path = self.directory / 'cities.csv'
        path.write_text('city,temperature\nBeijing,21\n', encoding='utf-8')
        response = self.parser.call({'url': str(path)})
        self.assertIn('| city', response)
        self.assertIn('Beijing', response)


if __name__ == '__main__':
    unittest.main()
