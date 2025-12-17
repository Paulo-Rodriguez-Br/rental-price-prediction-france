from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class RentalScraper:
    """Scraper for rental properties from the locamoi.fr website."""

    general_url: str = "https://locamoi.fr/location"
    url_suffixes: List[str] = field(default_factory=list)

    soup: Optional[BeautifulSoup] = None
    
    rows: List[Dict[str, Optional[str]]] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None
    
    session: requests.Session = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize output/log directories, logger, and HTTP session with retries.
        
        This method:
        - creates an output directory if it does not exist
        - creates a local logs/ directory if it does not exist
        - configures logging to logs/scraper.log
        - initializes a persistent requests.Session with retry adapters
        """
        BASE_DIR = Path(__file__).resolve().parent
        output_dir = BASE_DIR / "scraping_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "scraper.log"

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            filemode="a",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized...")
        
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)   
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.logger.info("Requests session initialized with retry adapters.")

    def fetch_soup(self, url: str) -> None:
        """
        Fetch HTML content from `url` and update `self.soup`.
    
        - Performs a GET request using the session configured with retry logic.
        - Network-level exceptions (timeouts, connection errors) may bubble up and
          must be handled by the caller.
        - If the server returns a non-200 HTTP status, the response is logged and
          `self.soup` is set to an empty BeautifulSoup instance so that downstream
          code can safely run CSS selectors without raising errors.
        - On successful responses (HTTP 200), `self.soup` is populated with the parsed HTML.
        """
        self.logger.debug("Fetching URL: %s", url)


        response = self.session.get(
            url,
            timeout=(10, 40),
        )
        
        if response.status_code != 200:
            self.logger.warning("Non-200 status for %s: %s", url, response.status_code)
            self.soup = BeautifulSoup("", "html.parser")
            
            return

        html = response.text
        self.soup = BeautifulSoup(html, "html.parser")

        self.logger.debug(
            "Fetched %s (status=%s, bytes=%s)",
            url,
            response.status_code,
            len(html),
        )

    def get_url_suffixes(self) -> None:
        """
        Populate url_suffixes with all rental detail URL suffixes found by
        iterating through paginated listing pages, segmented by rent ranges.
        """
        rent_step = 1
        current_min = 1
        rise_rent_step = True
        sped_up_rent_step = 1000
        limit_to_rise_rent_step = 5000
        highest_price = 40000

        while True:
            if rise_rent_step and current_min >= limit_to_rise_rent_step:
                rent_step = sped_up_rent_step
                rise_rent_step = False

                self.logger.info(
                    "Reached %s€+, speeding up rent_step to %s",
                    limit_to_rise_rent_step,
                    sped_up_rent_step,
                )

            current_max = current_min + rent_step - 1
            page_index = 1
            empty_pages = 0

            print(
                f"\n💶 Scanning rent range : {current_min}€ - {current_max}€... 💶\n"
            )
            self.logger.info(
                "Scanning rent range: %s€ - %s€",
                current_min,
                current_max,
            )

            while True:
                print(f"Fetching page {page_index}...")
                self.logger.debug(
                    "Fetching page %s for rent=%s-%s",
                    page_index,
                    current_min,
                    current_max,
                )

                if page_index == 1:
                    url = (
                        f"{self.general_url}"
                        f"?rent={current_min}-{current_max}"
                    )
                else:
                    url = (
                        f"{self.general_url}"
                        f"?rent={current_min}-{current_max}&page={page_index}"
                    )

                self.fetch_soup(url)

                try:
                    property_not_found = self.soup.select_one(
                        'body > div.xl\:mx-16 > div > div > div > h2'
                    ).text
                except Exception:
                    property_not_found = False

                if property_not_found == "Aucun bien trouvé":
                    print(
                        "\n🏁 No properties found. "
                        "Moving to the next range. 🏁\n"
                    )
                    
                    if page_index == 1:
                        self.logger.info(
                            "No properties found for rent range %s-%s. Moving on.",
                            current_min,
                            current_max,
                        )
                        
                    break

                tags = self.soup.select(
                    'div[data-testid="propertyTile"] a'
                )
                if not tags:
                    empty_pages += 1
                    print(f"\nEmpty pages: {empty_pages} out of 2\n")
                    self.logger.warning(
                        "Empty page %s/2 for rent range %s-%s (page_index=%s)",
                        empty_pages,
                        current_min,
                        current_max,
                        page_index,
                    )

                    if empty_pages >= 2:
                        print(
                            "🛑 Two empty pages limit reached. "
                            "Moving to the next range. 🛑\n\n"
                        )
                        self.logger.info(
                            "Two empty pages reached for rent range %s-%s. Moving on.",
                            current_min,
                            current_max,
                        )
                        break

                    page_index += 1
                    continue

                empty_pages = 0
                page_index += 1

                self.url_suffixes.extend(tag["href"] for tag in tags)
                self.logger.debug(
                    "Collected %s suffixes so far (latest batch size=%s)",
                    len(self.url_suffixes),
                    len(tags),
                )

            current_min = current_max + 1
            if current_min > highest_price:
                self.logger.info(
                    "Reached highest_price=%s. Stopping suffix scan.",
                    highest_price,
                )
                break

        self.url_suffixes = list(set(self.url_suffixes))
        self.logger.info(
            "Finished suffix scan. Unique suffixes=%s",
            len(self.url_suffixes),
        )

    @staticmethod
    def build_full_url(suffix: str) -> str:
        """
        Build a full property URL from a URL suffix.
        """
        return "https://locamoi.fr" + suffix

    def safe_select_text(self, selector: str) -> Optional[str]:
        """
        Safely select the text of the first element matching a CSS selector.
        Returns None if any error occurs or the element is not found.
        """
        if self.soup is None:
            return None
        try:
            tag = self.soup.select_one(selector)
            return tag.get_text(strip=True) if tag else None
        except Exception:
            return None

    def get_rental_data(self, url: str) -> Dict[str, Optional[str]]:
        """
        Extract rental data from a property detail page and return it as a dict.
        """
        self.logger.debug("Scraping rental data from: %s", url)

        self.fetch_soup(url)
        if self.soup is None:
            
            return {"Address": None}

        address = self.safe_select_text('a[href="#propertyMap"] p')

        rental_dict_data: Dict[str, Optional[str]] = {
            "Address": address,
        }

        try:
            selector_property_data = "div.flex.justify-between p"
            property_data = self.soup.select(selector_property_data)[0:10]
        except Exception:
            property_data = []

        for i in range(0, len(property_data), 2):
            try:
                key = property_data[i].get_text(strip=True)
                value = property_data[i + 1].get_text(strip=True)
                rental_dict_data[key] = value
            except Exception:
                continue

        self.logger.debug(
            "Extracted keys=%s from %s",
            list(rental_dict_data.keys()),
            url,
        )
        return rental_dict_data
        
    def run_scraping(self, resume: bool = False, start_index: int = 0) -> None:
        """
        Run the scraping process.
        
        If resume=False (default):
            - Collect URL suffixes
            - Scrape all properties from scratch
        
        If resume=True:
            - url_suffixes must already be populated
            - Resume scraping from start_index
        
        Results are accumulated in self.rows and converted to self.dataframe at the end.
        """
        if not resume:
            self.logger.info("Starting scraping run.")
            
            self.get_url_suffixes()
            
            self.logger.info("Starting properties scraping run.")
            print("\nStarting properties scraping run.\n")
            start_index = 0
            total = len(self.url_suffixes)
    
        else:
            total = len(self.url_suffixes)
            if start_index < 0 or start_index >= total:
                raise ValueError(
                    f"start_index must be between 0 and {total - 1}, got {start_index}"
                )
    
            self.logger.info(
                "Resuming property scraping from %s/%s",
                start_index + 1,
                total,
            )
            print(f"\nResuming scraping from index {start_index} of {total}\n")
    
        for counter in range(start_index, total):
            suffix = self.url_suffixes[counter]
    
            print(f"\nScraping progress: {counter + 1}/{total}")
            self.logger.info(
                "Scraping progress: %s/%s",
                counter + 1,
                total,
            )
    
            url = self.build_full_url(suffix)
    
            try:
                rental_dict_data = self.get_rental_data(url)
    
                if rental_dict_data.get("Address") is None and len(rental_dict_data) == 1:
                    self.logger.warning(
                        "Empty or invalid data for URL %s at page %s. Skipping.",
                        url, counter + 1
                    )
                    
                    continue
    
                self.rows.append(rental_dict_data)
    
            except Exception as e:
                self.logger.warning(
                    "Failed to process URL %s at page %s. Skipping. Details: %s",
                    url, counter + 1, e
                )
                
                continue
    
        self.dataframe = pd.DataFrame(data=self.rows)
        
        self.logger.info(
            "Scraping finished. Total rows=%s",
            len(self.dataframe) if self.dataframe is not None else 0,
        )
        print("\n🏁 Scraping finished.\n")
        

if __name__ == "__main__":
    
    start_script = input(
        "Do you want to start the scraping process from the beginning?\n"
        "Type YES to confirm, or press any other key to cancel: "
    )
    
    if start_script.lower() == 'yes':
        scraper = RentalScraper()
        scraper.run_scraping()

        BASE_DIR = Path(__file__).resolve().parent
        output_path = BASE_DIR / "scraping_outputs" / "rental_database.parquet"
        scraper.dataframe.to_parquet(output_path)
        
                
