import logging

def getLog(log_path, screen=False):
    if screen:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # to file
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # to screen
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # two Handler
        logger.addHandler(ch)
        logger.addHandler(fh)
    else:
        logging.basicConfig(
            filename=log_path,
            filemode="w",
            format="%(asctime)s %(message)s",
            level=logging.DEBUG,
        )
        logger = logging.getLogger("tst")

    return logger


if __name__ == "__main__":
    logger = getLog(log_path, screen=True)
    logger.info("this is info message")
    logger.warning("this is warn message")
