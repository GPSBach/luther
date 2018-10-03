'''
trulia_scrape_spyder.py is a scrapy spyder that retrieves public records data from the trulia database
https://www.trulia.com/property-sitemap
'''

import scrapy
from trulia_scrape.items import TruliaScrapeItem

class Trulia_Scrape(scrapy.Spider):
    name = 'trulia_scrape'
    allowed_domains = ['trulia.com']
    start_urls = [
    'https://www.trulia.com/property-sitemap/IL/Cook-County-17031/'
    ]


    def parse(self, response):
        root_domain = 'https://www.trulia.com'
        pages = response.xpath('/html/body/div[2]/div/div/div/div[2]/div/ul/li/a/@href').extract()
        while len(pages)!=0:
            item = Trulia_ScrapeItem()
            item['zip_url'] = root_domain + pages[0].strip("[,]'")
            request = scrapy.Request(item['zip_url'], callback = self.parseStreetData)
            request.meta['TruliaScrapeItem'] = item
            del(pages[0])
            yield request

    def parseStreetData(self, response):
        pages = response.xpath('_______').extract()
        while len(pages)!=0:
            item = TruliaScrapeSpiderItem()
            item['street_url'] = root_domain + pages[0].strip("[,]'")
            request = scrapy.Request(item['street_url'], callback = self.parseUnitData)
            request.meta['TruliaScrapeItem'] = item
            del(pages[0])
            yield request

    def parseUnitData(self, response):
        item = response.meta['TruliaScrapeItem']
        item = self.getUnitInfo(item, response)
        return item

    def getUnitInfo(self, item, response):
        item['address'] = response.xpath('_______').extract()
        return item
