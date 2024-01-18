from confoundcontinuum.logging import configure_logging, log_versions, logger


configure_logging()
log_versions()
logger.warning('Warning')
logger.info('Info')
logger.debug('Debug')