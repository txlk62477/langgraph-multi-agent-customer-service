"""MySQL 房源导入器的离线解析测试。"""

import importlib.util
from pathlib import Path
import unittest


IMPORTER_PATH = (
    Path(__file__).parents[3] / "借鉴" / "sql" / "import_bitehouse_to_postgres.py"
)
SPEC = importlib.util.spec_from_file_location("bitehouse_importer", IMPORTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载房源导入器")
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


class BitehouseImporterTests(unittest.TestCase):
    def test_parses_all_mysql_house_rows(self) -> None:
        source = IMPORTER_PATH.with_name("bitehouse_prd.sql")
        rows = list(importer.iter_house_rows(source))

        self.assertEqual(len(rows), 8912)
        self.assertTrue(all(len(row) == 23 for row in rows))
        self.assertEqual(rows[0][0], 11000010001)
        self.assertEqual(rows[-1][0], 64010010188)
        self.assertTrue(any("O'PARK" in row[11] for row in rows))


if __name__ == "__main__":
    unittest.main()
