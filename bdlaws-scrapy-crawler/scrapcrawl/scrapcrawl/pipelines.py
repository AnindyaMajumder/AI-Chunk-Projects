# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy import signals
from scrapy.exporters import CsvItemExporter, JsonItemExporter
from collections import defaultdict


class ScrapCrawlPipeline(object):
    def __init__(self):
        self.files = defaultdict(list)
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def spider_opened(self, spider):
        csv_file = open('%s.csv' % spider.name, 'w+b')
        json_file = open('%s.json' % spider.name, 'w+b')

        self.files[spider].append(csv_file)
        self.files[spider].append(json_file)

        self.exporters = [
            JsonItemExporter(json_file),
            CsvItemExporter(csv_file)
        ]

        for exporter in self.exporters:
            exporter.start_exporting()

    def spider_closed(self, spider):
        for exporter in self.exporters:
            exporter.finish_exporting()

        files = self.files.pop(spider)
        for file in files:
            file.close()

    def process_item(self, item, spider):
        for exporter in self.exporters:
            exporter.export_item(item)
        return item
