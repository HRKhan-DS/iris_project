#src/logger.py

import logging
import os
from datetime import datetime


log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir,exist_ok=True)

log_file_name= "{datetime.now().strftime('%Y-%m-%d-%H-%M-S').log}"
log_file_path=os.path.join(log_dir,log_file_name)

logging.basicConfig(
      filename=log_file_path,
      format='%(asctime)s -%(lineno)s -%(name)s -%(message)s',
      level=logging.INFO
                    )

if __name__=='__main__':
      logging.info('Logging start sucessfull')