scrapy_trulia subdirectory: scrapy codes for trulia.com
Because the bot blocker on trulia is extremely aggressive, this scraper requires the use of a proxy rotator like Crawlera.  Success rate ~30-50%.

Manage runs with:
Scraper executables: ./trulia_scraper/spiders/sale_XXX.py
Scraper output management: ./trulia_scraper/items.py
Scraper config files: ./trulia_scraper/settings.py & ./scrapy.cfg
Output management:

./transit_timings.py: retrieve and initially parse public transit times for each listing from google's directions API.  Also combines in mean and median income (by zipcode) from the US census bureau.

./second_parse.py
Secondary parse of output from google's directions API.  Separated from transit_timings.py intentionally, to reduce the chance of error during $/query api call.

linear_regression.py
Takes combined output from trulia, census bereau, and google distance API data, and predicts house price based on 7 variables.

