import pandas as pd
import time
import re
import logging
from typing import Generator, Literal
from bs4 import BeautifulSoup
from pandas import DataFrame, Series
from utilities.utils import fetch
import urllib3
from utilities.webdriver import Webdriver
urllib3.disable_warnings()
logger = logging.getLogger('default')


def geoGrab(address):
  from geopy.geocoders import Nominatim
  print(f"{address=}")
  geolocator = Nominatim(user_agent='user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36')
  location = geolocator.geocode(address)
  if location is not None:
    loc = {}
    loc['latitude'] = location.latitude
    loc['longitude'] = location.longitude
    return loc
  else:
    return None


def get_address(row: Series) -> str:
  city = row['city']
  ditrict = row['district']
  house_number = row['house_number']
  intersection = row['intersection']
  street = row['street']
  add = [
      f"{city}",
      f", {ditrict} район" if type(ditrict) != float else '',
      (f", {street}" if type(street) != float else ''),
      f" {house_number}" if type(house_number) != float else '',
      f" - {intersection}" if type(intersection) != float else '',
  ]
  # print(city, ditrict, house_number, intersection, street,)
  res = ''.join(add)
  res = re.sub(r'(\, (мкр|Мкр|Мкрн))?', '', res)
  return res


def placeFind(df: DataFrame):
  rows_list = []
  for ind, row in df.iterrows():
    address = get_address(row)

    coords = geoGrab(address)
    if coords is not None and 43 < coords['latitude'] < 44 and 76 < coords['longitude'] < 77:
      rows_list.append([address, coords['latitude'], coords['longitude']])
      print(f"{coords=}\n")
    else:
      rows_list.append([address, None, None])
  df2 = pd.DataFrame(rows_list, columns=['address', 'lat', 'long'])
  df3 = df.join(df2)
  df3.to_csv('df3.csv')


class Parser:
  def __init__(self, use_webdriver: bool = False) -> None:
    self.use_webdriver = use_webdriver
    if use_webdriver:
      chrome_driver_path = 'C:\\chromedriver.exe'
      firefox_binary_path = 'C:\\Users\\barlybay.kaisar\\Desktop\\Tor Browser\\Browser\\firefox.exe'
      firefox_profile_path = 'C:\\Users\\barlybay.kaisar\\Desktop\\Tor Browser\\Browser\\TorBrowser\Data\\Browser\\profile.default'
      geckodriver_path = 'C:\\geckodriver.exe'
      self.webdriver = Webdriver('tordriver', chrome_driver_path, firefox_binary_path, firefox_profile_path, geckodriver_path, True)
    self.dtypes = {
        'room_count': 'int64',
        'neighborhood': 'str',
        'street': 'str',
        'house_number': 'str',
        'intersection': 'str',
        'district': 'str',
        'floor': 'int64',
        'residential_complex': 'str',
        'max_floor': 'int64',
        'general_area': 'float64',
        'bathroom': 'bool',
        'living_area': 'float64',
        'kitchen_area': 'float64',
        'condition': 'str',
        'internet': 'bool',
        'furniture': 'bool',
        'ceiling_height': 'float64',
        'floor_type': 'str',
        'telephone': 'bool',
        'door': 'str',
        'balcony': 'bool',
        'parking': 'bool',
        'is_balcony_glazed': 'bool',
        'bars_on_the_window': 'bool',
        'security': 'bool',
        'entry_phone': 'bool',
        'code_lock': 'bool',
        'alarm': 'bool',
        'video_security': 'bool',
        'video_entry_phone': 'bool',
        'concierge': 'bool',
        'plastic_windows': 'bool',
        'non_angular': 'bool',
        'improved': 'bool',
        'rooms_isolated': 'bool',
        'studio_kitchen': 'bool',
        'kitchen_builtin': 'bool',
        'new_plumbing': 'bool',
        'pantry': 'bool',
        'counters': 'bool',  # ?
        'quiet_courtyard': 'bool',
        'air_conditioning': 'bool',
        'commercial_convenient': 'bool',
        'installment': 'bool',
        'mortgage': 'bool',
        'building_type': 'str',
        'build_year': 'int64',
        'price': 'int64',
        'mortgaged': 'bool',
        'images_count': 'int64',
        'private_hostel': 'bool',
        'city': 'str',
        'text': 'str',
    }

  def crawl(self, from_page: int, to_page: int) -> Generator[str, None, None]:
    for i in range(from_page, to_page + 1):
      url = f'https://krisha.kz/prodazha/kvartiry/almaty/?page={i}'
      logger.debug(url)
      soup = self.get_soup(url)
      a_tags = soup.select('div.a-card__header > div.a-card__main-info > div.a-card__header-left > a')

      for a_tag in a_tags:
        uri = f"https://krisha.kz/{a_tag['href']}"
        yield uri

  def get_soup(self, url: str):
    time.sleep(0.5)
    if self.use_webdriver:
      self.webdriver.driver.get(url)
      html = self.webdriver.driver.page_source
      return BeautifulSoup(html, 'html.parser')
    else:
      resp = fetch(url)
      return BeautifulSoup(resp.text, 'html.parser')

  def get_price(self, soup: BeautifulSoup) -> int:
    selector = 'div.offer__price'
    mortgage_selector = 'div.offer__sidebar-header > p'
    try:
      value = soup.select_one(selector).getText()
      return int(''.join(re.findall(r'\d+', value)))
    except AttributeError:
      value = soup.select_one(mortgage_selector).getText()
      return int(''.join(re.findall(r'\d+', value)))

  def get_mortgaged(self, soup: BeautifulSoup) -> bool:
    try:
      selector = 'div.offer__parameters-mortgaged'
      value = soup.select_one(selector).getText()
      return True
    except AttributeError as e:
      return False

  def get_title_info(self, soup: BeautifulSoup) -> int:
    title_info = {}
    selector = 'div.offer__advert-title > h1'
    value = soup.select_one(selector).getText().strip()
    pattern = (
        r'(?P<room_count>\d+)-.*'
        r'(, (?P<area>\d+\.?\d+) м²)'
        r'(, (?P<floor>\d+)/(?P<max_floor>\d+) этаж)?'
        r'(, мкр (?P<neighborhood>\w+(-\d+)?))?'
        r'(, (?P<street>\w+))?'
        r'( (?P<house_number>\d+(\w+)?(/)?(\s\w+)?(\d+)?))?'
        r'( — (?P<intersection>.*))?'
    )
    val = re.match(pattern, value).groupdict()
    title_info['room_count'] = int(val['room_count'])
    title_info['neighborhood'] = val['neighborhood']
    title_info['street'] = val['street']
    title_info['house_number'] = val['house_number']
    title_info['intersection'] = val['intersection']
    return title_info

  def get_others(self, soup: BeautifulSoup) -> int:
    others = {
        'plastic_windows': None,
        'non_angular': None,
        'improved': None,
        'rooms_isolated': None,
        'studio_kitchen': None,
        'kitchen_builtin': None,
        'new_plumbing': None,
        'pantry': None,
        'counters': None,
        'quiet_courtyard': None,
        'air_conditioning': None,
        'commercial_convenient': None,
    }
    selector = 'div.offer__description > div.text > div.a-options-text.a-text-white-spaces'
    try:
      value = soup.select_one(selector).getText().strip()
      vals = value.split(', ')
      for val in vals:
        val = re.sub(r'\.', '', val)
        match val.lower():
          case 'пластиковые окна':
            others['plastic_windows'] = True
          case 'неугловая':
            others['non_angular'] = True
          case 'улучшенная':
            others['improved'] = True
          case 'комнаты изолированы':
            others['rooms_isolated'] = True
          case 'кухня-студия':
            others['studio_kitchen'] = True
          case 'встроенная кухня':
            others['kitchen_builtin'] = True
          case 'новая сантехника':
            others['new_plumbing'] = True
          case 'кладовка':
            others['pantry'] = True
          case 'счётчики':
            others['counters'] = True
          case 'тихий двор':
            others['quiet_courtyard'] = True
          case 'кондиционер':
            others['air_conditioning'] = True
          case 'удобно под коммерцию':
            others['commercial_convenient'] = True
          case _:
            pass
    except AttributeError as e:
      logger.error("No others section")
    return others

  def get_private_hostel(self, soup: BeautifulSoup) -> dict[str, any]:
    for i in range(1, 10):
      selector1 = f'div.offer__parameters > dl:nth-child({i})'
      selector2 = f'div.offer__parameters > dl:nth-child({i}) > dd'
      try:
        value = soup.select_one(selector1).getText().strip()
        value = re.match(r'В прив. общежитии\.*', value)
        if value:
          try:
            value = soup.select_one(selector2).getText().strip()
          except AttributeError:
            return False
          return value == 'да'
      except AttributeError:
        continue
    return False

  def get_building_type__building_year(self, soup: BeautifulSoup) -> tuple[str, int]:
    selector = 'div:nth-child(2) > div.offer__advert-short-info'
    value = soup.select_one(selector).getText().strip()
    logger.debug(value)
    pattern = (
        r"(?P<building_type>\w+)?"
        r"(, )?"
        r"((?P<building_year>\d+) г.п.)?"
    )
    val = re.match(pattern, value).groupdict()
    return val['building_type'], int(val['building_year']) if val['building_year'] is not None else None

  def get_offer_short_description(self, soup: BeautifulSoup) -> tuple[float, float | None, float | None]:
    patterns: dict[str, dict[Literal['title_pattern'], str]] = {
        'areas': {'title_pattern': r'Площадь', },
        'floor_max_floor': {'title_pattern': r'Этаж'},
        'residential_complex': {'title_pattern': r'Жилой комплекс'},
        'district': {'title_pattern': r'Город'},
        'condition': {'title_pattern': r'Состояние'},
        'bathroom': {'title_pattern': r'Санузел'},
    }
    offer_short_description = {
        'floor': None,
        'max_floor': None,
        'residential_complex': None,
        'district': None,
        'condition': None,
        'bathroom': None,
    }
    for i in range(1, 10):
      selector1 = f"div:nth-child({i}) > div.offer__info-title"
      selector2 = f'div:nth-child({i}) > div.offer__advert-short-info'
      for param, params in patterns.items():
        try:
          title = soup.select_one(selector1).getText().strip()
          title_value = re.match(params['title_pattern'], title)
          if title_value:
            value = soup.select_one(selector2).getText().strip()
            match param:
              case 'areas':
                pattern = (
                    r"(?P<general_area>\d*\.?\d*) м²"
                    r"(, жилая — (?P<living_area>\d*\.?\d*)? м²)?"
                    r"(, кухня — (?P<kitchen_area>\d*\.?\d*)? м²)?"
                )
                val = re.match(pattern, value).groupdict()
                offer_short_description['general_area'] = float(val['general_area'])
                offer_short_description['living_area'] = float(val['living_area']) if val['living_area'] is not None else None
                offer_short_description['kitchen_area'] = float(val['kitchen_area']) if val['kitchen_area'] is not None else None
                # 148                 м², жилая — 88.9                  м², кухня — 24.7                   м²
              case 'floor_max_floor':
                try:
                  pattern = (
                      r"(?P<floor>\d+)"
                      r"( из (?P<max_floor>\d+))?"
                  )
                  val = re.match(pattern, value).groupdict()
                  offer_short_description['floor'] = int(val['floor'])
                  offer_short_description['max_floor'] = int(val['max_floor']) if val['max_floor'] is not None else None
                except AttributeError:
                  logger.error(f"{value}")
              case 'residential_complex':
                offer_short_description['residential_complex'] = value
              case 'bathroom':
                offer_short_description['bathroom'] = value
              case 'district':
                pattern = (
                    r"(?P<city>\w+)"
                    r"(, (?P<district>\w+) р-н)?"
                    r"\.*"
                )
                val = re.match(pattern, value).groupdict()
                offer_short_description['district'] = val['district']
              case 'condition':
                offer_short_description['condition'] = value
              case _:
                pass
        except AttributeError as e:
          continue
    # 45 м², кухня — 6 м²

    return offer_short_description

  def get_offer_description(self, soup: BeautifulSoup) -> tuple[float, float | None, float | None]:
    patterns: dict[str, dict[Literal['title_pattern'], str]] = {
        'telephone': {'title_pattern': r'Телефон', },
        'internet': {'title_pattern': r'Интернет', },
        'balcony': {'title_pattern': r'Балкон$', },
        'is_balcony_glazed': {'title_pattern': r'Балкон остеклён', },
        'door': {'title_pattern': r'Дверь', },
        'parking': {'title_pattern': r'Парковка', },
        'furniture': {'title_pattern': r'Мебель', },
        'floor_type': {'title_pattern': r'Пол$', },
        'ceiling_height': {'title_pattern': r'Потолки', },
        'security': {'title_pattern': r'Безопасность', },
        # 'internet': {'title_pattern': r'Санузел', },
    }
    offer_short_description = {
        'telephone': None,
        'internet': None,
        'balcony': None,
        'door': None,
        'parking': None,
        'is_balcony_glazed': None,
        'furniture': None,
        'floor_type': None,
        'ceiling_height': None,

        'bars_on_the_window': None,
        'security': None,
        'entry_phone': None,
        'code_lock': None,
        'alarm': None,
        'video_security': None,
        'video_entry_phone': None,
        'concierge': None,
    }
    for i in range(1, 20):
      selector1 = f"div.offer__description > div.offer__parameters > dl:nth-child({i}) > dt"
      selector2 = f"div.offer__description > div.offer__parameters > dl:nth-child({i}) > dd"
      for param, params in patterns.items():
        try:
          title = soup.select_one(selector1).getText().strip()
          title_match = re.match(params['title_pattern'], title)
          if title_match:
            value = soup.select_one(selector2).getText().strip()
            match param:
              case 'telephone':
                offer_short_description['telephone'] = value
              case 'internet':
                offer_short_description['internet'] = value
              case 'balcony':
                offer_short_description['balcony'] = value
              case 'is_balcony_glazed':
                offer_short_description['is_balcony_glazed'] = value
              case 'door':
                offer_short_description['door'] = value
              case 'parking':
                offer_short_description['parking'] = value
              case 'furniture':
                offer_short_description['furniture'] = value
              case 'floor_type':
                offer_short_description['floor_type'] = value
              case 'ceiling_height':
                val = re.match(r'(?P<ceiling_height>\d+\.?\d*) м', value)
                offer_short_description['ceiling_height'] = float(val['ceiling_height'])
              case 'security':
                vals = value.split(', ')
                for val in vals:
                  match val:
                    case 'решетки на окнах':
                      offer_short_description['bars_on_the_window'] = True
                    case 'охрана':
                      offer_short_description['security'] = True
                    case 'домофон':
                      offer_short_description['entry_phone'] = True
                    case 'кодовый замок':
                      offer_short_description['code_lock'] = True
                    case 'сигнализация':
                      offer_short_description['alarm'] = True
                    case 'видеонаблюдение':
                      offer_short_description['video_security'] = True
                    case 'видеодомофон':
                      offer_short_description['video_entry_phone'] = True
                    case 'консьерж':
                      offer_short_description['concierge'] = True
                    case _:
                      pass
              case _:
                pass
        except AttributeError as e:
          continue
    # 45 м², кухня — 6 м²

    return offer_short_description

  def get_installment_mortgage(self, soup: BeautifulSoup) -> tuple[bool, bool]:

    mortgage_selector = 'span.credit-badge.credit-badge--hypothec-full'
    installment_selector = 'span.credit-badge.credit-badge--installment'
    try:
      value = soup.select_one(mortgage_selector).getText().strip()
      mortgage = True
    except AttributeError as e:
      mortgage = False
    try:
      value = soup.select_one(installment_selector).getText().strip()
      installment = True
    except AttributeError as e:
      installment = False
    return installment, mortgage

  def get_city(self, soup: BeautifulSoup) -> str:
    return 'Алматы'

  def get_text(self, soup: BeautifulSoup) -> str:
    selector = r'div.offer__description > div.text'
    try:
      return soup.select_one(selector).getText().strip()
    except AttributeError as e:
      return None

  def get_images_count(self, soup: BeautifulSoup) -> str:
    selector = 'div.gallery__container > ul > li'
    image_lis = soup.select(selector)
    return len(image_lis)

  def scrape(self, uri: str) -> dict:
    logger.info(f"Scraping uri: {uri}")
    soup = self.get_soup(uri)
    # logger.debug(soup)
    params = {
        'uri': uri,
    }

    title_info = self.get_title_info(soup)
    params['room_count'] = title_info['room_count']
    params['neighborhood'] = title_info['neighborhood']
    params['street'] = title_info['street']
    params['house_number'] = title_info['house_number']
    params['intersection'] = title_info['intersection']

    offer_short_description = self.get_offer_short_description(soup)
    params['district'] = offer_short_description['district']
    params['floor'] = offer_short_description['floor']
    params['residential_complex'] = offer_short_description['residential_complex']
    params['max_floor'] = offer_short_description['max_floor']
    params['general_area'] = offer_short_description['general_area']
    params['bathroom'] = offer_short_description['bathroom']
    params['living_area'] = offer_short_description['living_area']
    params['kitchen_area'] = offer_short_description['kitchen_area']
    params['condition'] = offer_short_description['condition']

    offer_description = self.get_offer_description(soup)
    params['internet'] = offer_description['internet']
    params['furniture'] = offer_description['furniture']
    params['ceiling_height'] = offer_description['ceiling_height']
    params['floor_type'] = offer_description['floor_type']
    params['telephone'] = offer_description['telephone']
    params['door'] = offer_description['door']
    params['balcony'] = offer_description['balcony']
    params['parking'] = offer_description['parking']
    params['is_balcony_glazed'] = offer_description['is_balcony_glazed']

    params['bars_on_the_window'] = offer_description['bars_on_the_window']
    params['security'] = offer_description['security']
    params['entry_phone'] = offer_description['entry_phone']
    params['code_lock'] = offer_description['code_lock']
    params['alarm'] = offer_description['alarm']
    params['video_security'] = offer_description['video_security']
    params['video_entry_phone'] = offer_description['video_entry_phone']
    params['concierge'] = offer_description['concierge']

    others = self.get_others(soup)
    params['plastic_windows'] = others['plastic_windows']
    params['non_angular'] = others['non_angular']
    params['improved'] = others['improved']
    params['rooms_isolated'] = others['rooms_isolated']
    params['studio_kitchen'] = others['studio_kitchen']
    params['kitchen_builtin'] = others['kitchen_builtin']
    params['new_plumbing'] = others['new_plumbing']
    params['pantry'] = others['pantry']
    params['counters'] = others['counters']
    params['quiet_courtyard'] = others['quiet_courtyard']
    params['air_conditioning'] = others['air_conditioning']
    params['commercial_convenient'] = others['commercial_convenient']

    installment, mortgage = self.get_installment_mortgage(soup)
    params['installment'] = installment
    params['mortgage'] = mortgage

    building_type, building_year = self.get_building_type__building_year(soup)
    params['building_type'] = building_type
    params['build_year'] = building_year

    params['price'] = self.get_price(soup)
    params['mortgaged'] = self.get_mortgaged(soup)
    params['images_count'] = self.get_images_count(soup)
    params['private_hostel'] = self.get_private_hostel(soup)
    params['city'] = self.get_city(soup)
    params['text'] = self.get_text(soup)

    return params
