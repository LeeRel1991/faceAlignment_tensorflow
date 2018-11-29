#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------------------
# version: v0.1                                         
# author: lirui                 
# license: Apache Licence 
# contact: r.li@bmi-tech.com
# project: VideoHandler
# function: 
# file: log.py
# time: 16-10-17 下午5:24
# ---------------------------------------------------------

import logging
import logging.handlers as logHandle
import sys

if int(sys.version[0]) > 2:
    from io import StringIO
else:
    from  StringIO import StringIO
import os


class Logger:
    def __init__(self, log_file='', logger_name=''):
        # 设置日志输出的格式
        self._handr_list = ["ConHdr", "BufHdr"]
        self._formatter = logging.Formatter('%(name)s [%(asctime)s] %(levelname)s %(message)s')
        self._logger = logging.getLogger(logger_name)  # 日志输出方法的句柄
        self._log_file = log_file

        self.buf = StringIO()
        # self.BufHdr = None#logging.StreamHandler(self.buf)
        # self.ConHdr = None#logging.StreamHandler()
        # self.FileHdr = None#logHandle.RotatingFileHandler(self.__log_file, mode='w', maxBytes=10 * 1024 * 1024,
        #                                              # backupCount=5)
        self._init_log_handler()

        # 设置日志级别
        self._logger.setLevel(logging.INFO)

    def _init_log_handler(self):
        if self._log_file:
            self._handr_list.append('FileHdr')

        for hld in self._handr_list:
            self._create_handler(hld)
            # map(self.__create_handler, self.__handr_list)  # 创建病添加不同类别的日志句柄

    def _create_handler(self, handler_type="console"):
        func_dict = {"ConHdr": self._create_cons_handler, "FileHdr": self._create_file_handler,
                     "BufHdr": self._create_buf_handler}
        func_dict[handler_type]()
        getattr(self, handler_type).setFormatter(self._formatter)
        self._logger.addHandler(getattr(self, handler_type))

    def _create_cons_handler(self):
        self.ConHdr = logging.StreamHandler()

    def _create_file_handler(self):
        self.FileHdr = logHandle.RotatingFileHandler(self._log_file, mode='w', maxBytes=10 * 1024 * 1024,
                                                     backupCount=5)

    def _create_buf_handler(self):

        self.BufHdr = logging.StreamHandler(self.buf)

    def close_log_handler(self):
        map(self._logger.removeHandler, map(lambda x: getattr(self, x), self._handr_list))

    def addLog(self, message, level='info'):
        """
        添加日志
        :param message:
        :param level: 日志级别，"debug","info","warning","error"
        :return:
        """
        getattr(self._logger, level, '')(message)
        # print self.buf.getvalue().split('\n')[-2]

    def setlevel(self, level):
        """
        :param level: "INFO","DEBUG","WARNING","ERROR"
        :return:
        """
        self._logger.setLevel(getattr(logging, level, ''))

    def getLog(self):
        mess = self.buf.getvalue()
        self.buf.seek(0)
        self.buf.truncate()
        return mess

    def set_formatter(self, formatter):
        self._formatter = logging.Formatter(formatter)
        map(lambda x: getattr(self, x).setFormatter(self._formatter), self._handr_list)


if __name__ == '__main__':
    logger = Logger('demo.log', 'demo')
    logger.setlevel("WARN")
    logger.addLog("akshdkl", "info")
    # logger.close_log_handler()
    # logger = Logger('demo.log', 'demo')
    logger.addLog("13123425", "warning")
