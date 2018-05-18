# -*- coding: utf-8 -*-
import scrapy
import re
import requests
import json
import os

class XimalayeSpider(scrapy.Spider):
    """
        This spider is used to crawl wave file at ximalaya website 
    
    """
    name = "ximalaya"
    allowed_domains = ["ximalaya.com"]
    start_urls = ['http://www.ximalaya.com/dq/']
    self.header = {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:59.0) Gecko/20100101 Firefox/59.0'}
    self.download_dir = '/home/gwq/audio/crawler/'

    def parse(self, response):
        """
        parse the start url page, recursive by the category.
        """
        
        # fllow links to each categary page.
        categary_list = response.xpath("//div[@id='discoverAlbum']/div[@class='layout_left']//ul[@class='sort_list']/li/a/@href").extract()[1:2]
        for url in categary_list:
            categary_url = response.url.replace(re.findall('/dq/.*[/]?$', response.url)[0], url)
            yield scrapy.Request(categary_url, self.parse_one_category)



    def parse_one_category(self, response):
        """
        parse category page.
        """
        
        # parse every album in this page
        album_urls_list = response.xpath("//div[@id='discoverAlbum']/div[@id='explore_album_detail_entry']//div[@class='discoverAlbum_item']//a[@class='discoverAlbum_title']/@href").extract()
        if album_urls_list:
            # parse the each album page sand download wav files
            for url in album_urls_list:
                yield scrapy.Request(url, callback=self.parse_one_album)
              
        
        # parse the next page of this category
        next_page_list = response.xpath("//div[@id='discoverAlbum']/div[@id='explore_album_detail_entry']//div[@class='pagingBar_wrapper']/a[@rel='next']")
        # if there exists a next page, then go on scrawl
        if next_page_list:
            # next_page is a selector, which can use xpath to find its attribution
            next_page = next_page_list[0]
            next_url = response.url.replace(re.findall('/dq/.*/\d*[/]?$' , response.url)[0], next_page.xpath('@href').extract_first())
            yield scrapy.Request(next_url, callback=self.parse_one_category)


    def parse_one_album(self, response):
        """
        parse album pagge
        """
        # get the sound id list of this album
        soundlist = response.xpath("//div[@class='personal_container']/div[@class='personal_body']/@sound_ids").extract_first()
        if soundlist:
            for soundId in soundlist.split(','):
                # get json url
                json_url = 'http://www.ximalaya.com/tracks/{}.json'.format(soundId)
                response_json = json.loads(requests.get(json_url, headers=self.header).text)
                wav_url = response_json['play_path_64']
                category = response_json['category_name']
                nickname = response_json['nickname']
                title = response_json['title']
                store_dir = self.download_dir+category+'/'+nickname+'/'
                try:
                    os.makedirs(store_dir)
                except:
                    pass
                store_path = store_dir + title + '_' + soundId + '.m4a'
                with open(store_path, 'w+') as f:
                    f.write(requests.get(wav_url).content)

class BeiguangSpider(scrapy.Spider):

    name = 'beijingguangbo'