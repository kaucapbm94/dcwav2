from utilities.logger import get_script_logger
import pandas as pd
import numpy as np

# from dcwav.project.parser import Parser
from parser import Parser


logger = get_script_logger('DEBUG')


if __name__ == '__main__':
  logger.debug('Hello world')
  parser = Parser()
  for i in range(0, 20):
    paramss = []

    # 1 - 2000
    from_page = 1 + i * 100
    to_page = (i+1) * 100

    for uri in parser.crawl(from_page, to_page):
      params = parser.scrape(uri)
      paramss.append(params)
    df = pd.DataFrame(paramss, columns=list(parser.dtypes.keys()))
    for column_name, dtype in parser.dtypes.items():
      logger.debug(f"{column_name}: {df[column_name].dtypes} - {dtype}")
      match dtype:
        case 'int64':
          df[column_name] = np.floor(pd.to_numeric(df[column_name], errors='coerce')).astype('Int64')
        case 'str':
          df[column_name] = df[column_name].astype('str')
        case 'float64':
          df[column_name] = df[column_name].astype('Float64')
        case 'bool':
          df[column_name] = df[column_name].astype('bool')
        case _:
          pass
    df.to_csv(f"krisha_{from_page}-{to_page}.csv", encoding='utf-8')
    print(df.head)
