import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import pypdf
import time
import io
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

visited_urls = set()
bad_urls = []

def is_url(url):
    try:
        result = urlparse(url)
        return True
    except ValueError:
        return False

# Function to save only the title and body text
def scrape_and_save(base_url, base_directory, soup, response, content_type):
    if 'application/pdf' in content_type:
        content = io.BytesIO(response.content)
        pdf = pypdf.PdfReader(content)
        text = ""
        for page_num in range(len(pdf.pages)):
            text += pdf.pages[page_num].extract_text()
    else:
        # Extract the title and body content
        title = soup.title.string if soup.title else 'No Title'
        body = soup.body.get_text(separator=' ', strip=True) if soup.body else 'No Body Text'

        # print the following text in red
        if title == 'No Title' and body == 'No Body Text':
            print(f"\033[91mFollowing file is completely empty : {base_url} \033[0m")

        # Combine title and body for saving
        text = f"Title: {title}\n\nBody: {body}"

    # Parse the URL to create a valid file name
    parsed_url = urlparse(base_url)
    file_name = parsed_url.path.strip('/').replace('/', '_') or 'home_page'
    file_path = os.path.join(base_directory, f"{file_name}.txt")

    # Write the title and body text to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def find_all_subpages(url, soup, content_type):
    if 'application/pdf' in content_type:
        return []
    
    links = []
    for a in soup.find_all('a', href=True):
        link = a['href']
        full_link = urljoin(url, link)
        if not is_url(full_link):
            continue
        link_domain = urlparse(full_link).netloc

        # Only collect internal links (to the same domain)
        if link_domain == urlparse(url).netloc:
            links.append(full_link)

    return links

def start_browser():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless if you don't need GUI
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-infobars")

    # Automatically download and set up ChromeDriver
    driver_path = '/Users/kgdhatchi/Downloads/chromedriver-mac-arm64/chromedriver'
    service = Service(driver_path)
    
    # Create the WebDriver
    browser = webdriver.Chrome(service=service, options=chrome_options)
    return browser

def scrape_website_selenium(base_url, base_directory, depth=1, max_depth=4):
    if base_url in visited_urls:
        return
    
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to retrieve {base_url}. Status code: {response.status_code}")
        bad_urls.append(base_url)
        visited_urls.add(base_url)
        return
    
    browser = start_browser()
    print(f"Scraping {base_url} using Selenium...")
    browser.get(base_url)

    # Wait for page to fully load, in case of JavaScript-heavy content
    # time.sleep(5)

    page_source = browser.page_source
    browser.quit()
    soup = BeautifulSoup(page_source, 'html.parser')

    scrape_and_save(base_url, base_directory, soup, 'None', 'None')
    visited_urls.add(base_url)

    if depth < max_depth:
        links = find_all_subpages(base_url, soup, 'None')

        for link in links:  
            if link not in visited_urls:
                scrape_website_selenium(link, base_directory, depth + 1, max_depth)    

def scrape_website(base_url, base_directory, depth=1, max_depth=4, cookie=None):
    if base_url in visited_urls:
        return
    
    HEADERS = {
        "User-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    
    response = requests.get(base_url, headers=HEADERS, cookies=cookie, verify=False)
    if cookie is None:
        cookie = response.cookies
    if response.status_code != 200:
        print(f"Failed to retrieve {base_url}. Status code: {response.status_code}")
        bad_urls.append(base_url)
        return
    print(f"Retrieving {base_url}.")
    content_type = response.headers.get('content-type')
    soup = BeautifulSoup(response.content, 'html.parser')


    scrape_and_save(base_url, base_directory, soup, response, content_type)
    visited_urls.add(base_url)
    
    if depth < max_depth:
        links = find_all_subpages(base_url, soup, content_type)

        for link in links:
            if link not in visited_urls:
                scrape_website(link, base_directory, depth + 1, max_depth, cookie)    

def parallel_scraping(base_urls, base_directory, num_workers=None):
    # Set the number of workers to match the number of CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for url in base_urls:
            domain = urlparse(url).netloc
            website_directory = os.path.join(base_directory, domain)

            if not os.path.exists(website_directory):
                os.makedirs(website_directory)

            print(f"Scraping {url} and its subpages...")

            futures.append(executor.submit(scrape_website, url, website_directory))
            
        for future in as_completed(futures):
            try:
                future.result()  # Collect the results to catch any exceptions
            except Exception as e:
                print(f"Error occurred: {e}")



if __name__ == "__main__":
    # List of websites to scrape
    base_urls = [
        'https://www.wikiwand.com/en/articles/Pittsburgh', 
        'https://www.wikiwand.com/en/articles/History_of_Pittsburgh',
        'https://www.britannica.com/place/Pittsburgh',
        'https://www.visitpittsburgh.com/',
        'https://pittsburghpa.gov/finance/tax-forms',
        'https://pittsburghpa.gov/index.html',
        'https://apps.pittsburghpa.gov/redtail/images/23255_2024_Operating_Budget.pdf',
        'https://www.cmu.edu/about/',
        'https://pittsburgh.events/',
        'https://downtownpittsburgh.com/events/',
        'https://events.cmu.edu/'
        'https://www.cmu.edu/engage/alumni/events/campus/index.html',
        'https://pittsburghopera.org/',
        'https://www.thefrickpittsburgh.org/',
        'https://www.wikiwand.com/en/articles/List_of_museums_in_Pittsburgh',
        'https://www.visitpittsburgh.com/events-festivals/food-festivals/',
        'https://www.pghtacofest.com/',
        'https://pittsburghrestaurantweek.com/',
        'https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/',
        'https://www.mlb.com/pirates',
        'https://www.steelers.com/',
        'https://www.nhl.com/penguins/',
        'https://carnegiemuseums.org/',
        'https://www.heinzhistorycenter.org/',
        'https://bananasplitfest.com/',
        'https://littleitalydays.com/',
        'https://www.picklesburgh.com/',
        'https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d',
        'https://www.pittsburghsymphony.org/',
        'https://trustarts.org/',
    ]
    
    # Base directory to save the scraped data
    base_directory = '../../data/scraped_data'

    # get the numer of workers
    num_workers = os.cpu_count()//2

    # chunk the base_urls into bins of num_workers
    for i in range(0, len(base_urls), num_workers):
        base_urls_chunk = base_urls[i:i+num_workers]
        parallel_scraping(base_urls_chunk, base_directory)

    for base_url in base_urls:
        domain = urlparse(base_url).netloc
        website_directory = os.path.join(base_directory, domain)

        if not os.path.exists(website_directory):
            os.makedirs(website_directory)

        scrape_website_selenium('https://events.cmu.edu/all', website_directory, 1, 25) 
    
    # write the bad urls to a file
    with open('../../data/bad_urls.txt', 'w') as file:
        for url in bad_urls:
            file.write(f"{url}\n")
    
    print(f"\033[92mScraping complete!!\033[0m")