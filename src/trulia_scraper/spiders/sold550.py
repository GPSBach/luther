# -*- coding: utf-8 -*-
import os
import scrapy
from scrapy.linkextractors import LinkExtractor
import trulia_scraper.parsing as parsing
from trulia_scraper.items import TruliaItem, TruliaItemLoader
import trulia_scraper.spiders.trulia as trulia
from scrapy.utils.conf import closest_scrapy_cfg


class TruliaSpider(scrapy.Spider):
    name = 'sold_550'
    allowed_domains = ['trulia.com']
    custom_settings = {'FEED_URI': os.path.join(os.path.dirname(closest_scrapy_cfg()), 'data/iterate/sold_%(start)s_%(time)s.jl'), 
                       'FEED_FORMAT': 'jsonlines'}

    def __init__(self, state = 'IL', city = 'Chicago', start = 550, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.city = city
        self.start = start
        self.start_urls = ['https://www.trulia.com/sold/{city},{state}/'.format(state=state, city=city)]
        self.le = LinkExtractor(allow=r'^https://www.trulia.com/p/')

    def parse(self, response):
        #N = 598 #trulia.TruliaSpider.get_number_of_pages_to_scrape(response)
        M = self.start
        N = M+50
        self.logger.info("Seaching between index page {M} and index page {N} ".format(N=N,M=M))
        for url in [response.urljoin("{n}_p/".format(n=n)) for n in range(M, N+1)]:
            yield scrapy.Request(url=url, callback=self.parse_index_page)

    def parse_index_page(self, response):
        for link in self.le.extract_links(response):
            yield scrapy.Request(url=link.url, callback=self.parse_property_page)

    def parse_property_page(self, response):
        item_loader = TruliaItemLoader(item=TruliaItem(), response=response)
        trulia.TruliaSpider.load_common_fields(item_loader=item_loader, response=response)

        details = item_loader.nested_css('.homeDetailsHeading')
        taxes = details.nested_xpath('.//*[text() = "Property Taxes and Assessment"]/parent::div')
        taxes.add_xpath('property_tax_assessment_year', './following-sibling::div/div[contains(text(), "Year")]/following-sibling::div/text()')
        taxes.add_xpath('property_tax', './following-sibling::div/div[contains(text(), "Tax")]/following-sibling::div/text()')
        taxes.add_xpath('property_tax_assessment_land', './following-sibling::div/div/div[contains(text(), "Land")]/following-sibling::div/text()')
        taxes.add_xpath('property_tax_assessment_improvements', './following-sibling::div/div/div[contains(text(), "Improvements")]/following-sibling::div/text()')
        taxes.add_xpath('property_tax_assessment_total', './following-sibling::div/div/div[contains(text(), "Total")]/following-sibling::div/text()')
        taxes.add_xpath('property_tax_market_value', './following-sibling::div/div[contains(text(), "Market Value")]/following-sibling::div/text()')

        item = item_loader.load_item()
        trulia.TruliaSpider.post_process(item=item)
        return item
