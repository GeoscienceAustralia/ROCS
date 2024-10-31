# Custom formatters for logging

import logging

class MixedFormatter(logging.Formatter):

    def format(self, record):
        if record.levelno is logging.WARNING:
            if 'in function' not in record.msg: # avoid adding duplicates to
                                                # the record.msg
                record.msg = "[%s] In %s in function %s:\n %s\n" % (
                    record.levelname, record.filename,
                    record.funcName, record.msg)
        elif record.levelno in (logging.DEBUG,
                                logging.ERROR,
                                logging.CRITICAL):
            if 'at line' not in record.msg: # avoid adding duplicates to the
                                            # record.msg
                record.msg = "[%s] In %s at line %s :\n %s\n" % (
                    record.levelname, record.filename,
                    record.lineno, record.msg)

        return super(MixedFormatter , self).format(record)