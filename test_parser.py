from typing import Literal
from .parser import Parser
from unittest import TestCase
from utilities.logger import get_script_logger
logger = get_script_logger('DEBUG')

building_types = ['монолитный', 'кирпичный', 'панельный', 'иное']

test_cases: dict[str, dict[str, tuple[any, Literal['equal', 'in']]]] = {
    'https://krisha.kz/a/show/674185165': {
        'mortgaged': (True, 'equal'),
        'building_type': (building_types, 'in'),
        'build_year': (1963, 'equal'),
        'floor': (2, 'equal'),
        'max_floor': (4, 'max_floor'),
        'general_area': (45, 'equal'),
        'kitchen_area': (6, 'equal'),
        'mortgage': (False, 'equal'),
        'installment': (False, 'equal'),
        'private_hostel': (False, 'equal'),
        'district': ('Бостандыкский', 'equal'),
        'neighborhood': ('Алмагуль', 'equal'),
        'street': ('Егизбаева', 'equal'),
        'house_number': ('12а', 'equal'),
        'intersection': ('Розыбакиева', 'equal'),
        'room_count': (2, 'equal'),
        'images_count': (14, 'equal'),
        'condition': ('хорошее', 'equal'),
        'bathroom': ('совмещенный', 'equal'),
    },
    'https://krisha.kz/a/show/665531675': {
        'general_area': (148, 'equal'),
        'living_area': (88.9, 'equal'),
        'kitchen_area': (24.7, 'equal'),
        'mortgage': (True, 'equal'),
        'installment': (True, 'equal'),
        'private_hostel': (False, 'equal'),
        'residential_complex': ('Lafayette', 'equal'),
        'street': ('Сейдимбека', 'equal'),
        'house_number': ('110/1', 'equal'),
    },
    'https://krisha.kz/a/show/673950872': {
        'private_hostel': (True, 'equal'),
        'neighborhood': ('Аксай-3', 'equal'),
        'house_number': ('10 А', 'equal'),
        'intersection': ('Толе би', 'equal'),
        'entry_phone': (True, 'equal'),
        'bars_on_the_window': (True, 'equal'),
        'video_security': (True, 'equal'),
        'plastic_windows': (True, 'equal'),
        'non_angular': (True, 'equal'),
        'quiet_courtyard': (True, 'equal'),
    },
    'https://krisha.kz/a/show/673740963': {
        'internet': ('оптика', 'equal'),
        'balcony': ('несколько балконов или лоджий', 'equal'),
        'is_balcony_glazed': ('да', 'equal'),
        'door': ('деревянная', 'equal'),
        'parking': ('паркинг', 'equal'),
        'furniture': ('частично меблирована', 'equal'),
        'floor_type': ('ламинат', 'equal'),
        'ceiling_height': (2.8, 'equal'),
        'entry_phone': (True, 'equal'),
    },
    'https://krisha.kz//a/show/672226546': {
        'ceiling_height': (3, 'equal'),
    },
    'https://krisha.kz//a/show/673410125': {
    },
}


# pytest -v -s dcwav/project/test_parser.py
class TestViews(TestCase):
  @classmethod
  def setUpClass(self) -> None:
    super(TestViews, self).setUpClass()
    self.parser = Parser()
    logger.info("up")

  @classmethod
  def tearDownClass(self) -> None:
    super(TestViews, self).tearDownClass()
    logger.info("down")

  # pytest -v -s dcwav/project/test_parser.py::TestViews::test_crawl
  def test_crawl(self) -> None:
    for uri in self.parser.crawl(1, 1):
      logger.debug(uri)

  # pytest -v -s dcwav/project/test_parser.py::TestViews::test_regex
  def test_regex(self) -> None:
    import re
    pattern = r"(?P<general_area>\d+) м²(, жилая — (?P<living_area>\d*\.?\d*)? м²)?"
    val = re.match(pattern, '148 м², жилая — 88.9 м², кухня — 24.7 м²').groupdict()
    logger.debug(val)

  # pytest -v -s dcwav/project/test_parser.py::TestViews::test_scrape
  def test_scrape(self) -> None:
    for uri, feat in test_cases.items():
      params = self.parser.scrape(uri)
      logger.debug(params)
      for param_name, test_case in feat.items():
        value = test_case[0]
        assert_type = test_case[1]
        match assert_type:
          case 'equal':
            self.assertEqual(params[param_name], value, {param_name: value})
          case 'in':
            self.assertIn(params[param_name], value, {param_name: value})
