# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Item, Field

class ScrapLawItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    url_id = scrapy.Field()
    law_title = scrapy.Field()
    vol_ordinance = scrapy.Field()
    law_pass_date = scrapy.Field()
    law_subtitle = scrapy.Field()
    law_descripton = scrapy.Field()
    section_chapter_id = scrapy.Field()
    section_chapter_name = scrapy.Field()
    section_chapter_no = scrapy.Field()
    section_id = scrapy.Field()
    section_name = scrapy.Field()
    section_description = scrapy.Field()
