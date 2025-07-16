# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
from scrapy.http import HtmlResponse
from scrapy.utils.python import to_bytes
from selenium import webdriver
from scrapy.utils.project import get_project_settings
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys

import time
import re

settings = get_project_settings()

HEADLESS = True


class ScrapschedSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class ScrapschedDownloaderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


# class SeleniumMiddleware(object):
#
#     @classmethod
#     def from_crawler(cls, crawler):
#         middleware = cls()
#         crawler.signals.connect(middleware.spider_opened, signals.spider_opened)
#         crawler.signals.connect(middleware.spider_closed, signals.spider_closed)
#         return middleware
#
#     def process_request(self, request, spider):
#         request.meta['driver'] = self.driver  # to access driver from response
#         # self.driver.get(request.url)
#         if re.match("https://twitter.com/[\w]+/followers", request.url):
#             # self.driver.find_element_by_xpath("//div[@class='stream-footer']")
#             # time_end = self.driver.find_element_by_xpath("//div[contains(@class,'timeline-end')]")
#             self.driver.get("https://www.twitter.com/login")
#
#             elem = self.driver.find_element_by_css_selector(".js-initial-focus")
#             elem.clear()
#             email = 'nahid002345@gmail.com'
#             elem.send_keys(email)
#
#             elem = self.driver.find_element_by_css_selector(".js-password-field")
#             elem.clear()
#             password = 'NahidUddin89'
#             elem.send_keys(password)
#
#             elem.send_keys(Keys.RETURN)
#             time.sleep(2)
#
#             self.driver.get(request.url)
#
#             time_end = ''
#             page_length = self.driver.execute_script(
#                 "window.scrollTo(0, document.body.scrollHeight);var page_length=document.body.scrollHeight;return page_length;")
#             match = False
#             page_count = 1
#             while (match == False):
#                 print('page : ' + str(page_count))
#                 page_count += 1
#                 last_page_count = page_length
#                 time.sleep(2)
#                 page_length = self.driver.execute_script(
#                     "window.scrollTo(0, document.body.scrollHeight);var page_length=document.body.scrollHeight;return page_length;")
#                 is_reload_tried = False
#
#                 if last_page_count == page_length:
#                     try:
#                         time.sleep(2)
#                         time_end = self.driver.find_element_by_xpath(
#                             "//div[@class ='timeline-end has-items has-more-items']")
#                         print(time_end.text)
#                         print("check end stream: " + str(time_end.text))
#                     except NoSuchElementException:
#                         match = True
#                         print('timeline-end class not found')
#                         pass
#
#             # while time_end is not None:
#             #     self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#             #     time.sleep(30)
#             #     time_end = self.driver.find_element_by_xpath("//div[contains(@class,'timeline-end')]")
#
#         body = to_bytes(self.driver.page_source)  # body must be of type bytes
#         return HtmlResponse(self.driver.current_url, body=body, encoding='utf-8', request=request)
#
#     def spider_opened(self, spider):
#         if HEADLESS:
#             options = Options()
#             options.add_argument("--headless")  # Runs Chrome in headless mode.
#             options.add_argument('--no-sandbox')  # Bypass OS security model
#             options.add_argument('--disable-gpu')  # applicable to windows os only
#             options.add_argument('start-maximized')  #
#             options.add_argument('disable-infobars')
#             options.add_argument("--disable-extensions")
#             self.driver = webdriver.Chrome(chrome_options=options,
#                                            executable_path=r'G:\Projects\uni\event-attendance-prediction\bin\crawl\chromedriver\chromedriver')
#
#     def spider_closed(self, spider):
#         self.driver.close()
